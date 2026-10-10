"""Unified trainer for the four-method comparison (SPEC.md §8).

  --method sft_good    : SFT on mode-correct rollouts (self-distillation)
  --method grpo        : GRPO/TTRL per-rollout advantage policy gradient
  --method echo        : ECHO entropy-confidence-shaped TTRL update
  --method es_direct   : Full-parameter Evolution Strategies (no backprop)
  --method es_direct_lora: Evolution Strategies over rank-16 LoRA only
  --method es_direct_lora_batch16: Same LoRA ES with all N adapters generated
                                   concurrently on one GPU
  --method es_gr_wdpo  : ES-guided Group-Relative Weighted DPO (ours)

Adapted here for Qwen3 Base and the MATH-500 -> MMLU -> MedQA continual
benchmark. All label-free methods share the same prompts, population, modal
pseudo-reward gate, and visit budget; only their update/reward rule differs.

Exploration (per SPEC §5): sft_good & grpo sample N rollouts from the *student*
(single-model sampling, as in TTRL); the ES methods use parameter perturbations.
"""
import argparse
import gc
import os
import random
import time

import torch
import torch.nn.functional as F

import common as C


def build_models(args, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # es_direct = faithful to the ES paper: FULL-parameter ES, no LoRA. The model
    # itself is the optimized object; ES perturbs & steps all weights in place.
    if args.method == "es_direct":
        # ES-direct checkpoints are complete models rather than LoRA adapters.
        # At a restart or continual-stage boundary, load the inherited full
        # checkpoint while retaining args.model for tokenizer/prompt metadata.
        model_source = args.init_adapter or args.model
        student = AutoModelForCausalLM.from_pretrained(
            model_source, torch_dtype=dtype).to(device)
        if args.init_adapter:
            print(f"[es_direct] resumed full model from {args.init_adapter}")
        for p in student.parameters():
            p.requires_grad_(False)            # ES needs no autograd
        return tok, student, None, None        # (tok, student, teacher, kl_ref)

    student_base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype).to(device)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student_base, lora).to(device)

    # --init-adapter: resume training from a previous snapshot's LoRA. Weights are
    # copied in MANUALLY from safetensors (PeftModel.from_pretrained/adapter
    # machinery has segfaulted repeatedly on this box; we only need the tensors).
    if args.init_adapter:
        from safetensors.torch import load_file
        ad = load_file(os.path.join(args.init_adapter, "adapter_model.safetensors"))
        sd = student.state_dict()
        loaded = 0
        with torch.no_grad():
            for k, v in ad.items():
                sk = k.replace("lora_A.weight", "lora_A.default.weight") \
                      .replace("lora_B.weight", "lora_B.default.weight")
                if sk in sd:
                    sd[sk].copy_(v.to(sd[sk].device, sd[sk].dtype))
                    loaded += 1
        if loaded == 0:
            raise RuntimeError(f"--init-adapter: no tensors matched from {args.init_adapter}")
        print(f"[{args.method}] resumed LoRA from {args.init_adapter} ({loaded} tensors)")

    if args.method == "es_direct_lora_batch16":
        # One frozen inference adapter per population member.  All share the
        # single Qwen3 base model; only the small LoRA A/B matrices are copied.
        # PEFT's adapter_names generation argument selects a different adapter
        # for each batch row, so all N ES rollouts advance concurrently.
        for member in range(args.population):
            student.add_adapter(f"es_member_{member}", lora)
        student.set_adapter("default")
        print(f"[es_direct_lora_batch16] installed {args.population} rollout adapters")

    # es_gr_wdpo/es_grpo: frozen ES teacher (perturbed + re-anchored).
    # student_dpo: frozen base model = the DPO reference pi_0 (never modified).
    teacher = None
    if args.method in ("es_gr_wdpo", "es_grpo", "student_dpo"):
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    # dedicated frozen base = pi_0 for the GRPO-style KL term (separate model so we
    # never toggle the PEFT adapter, which corrupts named_parameters in a loop)
    kl_ref = None
    if args.kl_lambda > 0:
        kl_ref = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype).to(device)
        kl_ref.eval()
        for p in kl_ref.parameters():
            p.requires_grad_(False)
    return tok, student, teacher, kl_ref


def direct_es_parameters(student, method):
    """Return the stable parameter list perturbed and updated by direct ES.

    Full ES traverses every model tensor. LoRA ES traverses only PEFT's
    trainable adapter tensors, leaving the Qwen3 base weights frozen.
    """
    if method == "es_direct":
        params = list(student.parameters())
    elif method in ("es_direct_lora", "es_direct_lora_batch16"):
        params = [parameter for parameter in student.parameters()
                  if parameter.requires_grad]
    else:
        raise ValueError(f"not a direct-ES method: {method}")
    if not params:
        raise RuntimeError(f"{method} selected no parameters for ES")
    return params


def batched_lora_es_parameters(student, population):
    """Return the central LoRA parameters and matching member-adapter tensors.

    PEFT names the checkpoint adapter ``default`` and every inference-only
    population adapter ``es_member_K``.  Matching by name prevents an update
    from ever touching the frozen Qwen3 base weights.
    """
    named = dict(student.named_parameters())
    central_named = [
        (name, parameter) for name, parameter in named.items()
        if parameter.requires_grad
    ]
    if not central_named:
        raise RuntimeError("batched LoRA ES found no trainable central parameters")
    unexpected = [name for name, _ in central_named if ".default." not in name]
    if unexpected:
        raise RuntimeError(
            "batched LoRA ES expected only default-adapter parameters; found "
            + ", ".join(unexpected[:3])
        )

    member_names = [f"es_member_{member}" for member in range(population)]
    member_parameter_sets = []
    for member_name in member_names:
        parameters = []
        for central_name, central_parameter in central_named:
            target_name = central_name.replace(
                ".default.", f".{member_name}.", 1
            )
            target = named.get(target_name)
            if target is None or target.shape != central_parameter.shape:
                raise RuntimeError(
                    f"missing LoRA ES member tensor {target_name!r}"
                )
            parameters.append(target)
        member_parameter_sets.append(parameters)
    return [parameter for _, parameter in central_named], member_names, member_parameter_sets


@torch.no_grad()
def load_batched_lora_perturbations(central_params, member_parameter_sets,
                                    seeds, sigma, device):
    """Materialize theta + sigma*epsilon_k into each inference adapter."""
    for seed, member_params in zip(seeds, member_parameter_sets):
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        for central, member in zip(central_params, member_params):
            noise = torch.randn(
                central.shape, generator=generator, device=device,
                dtype=torch.float32,
            )
            member.copy_(central + sigma * noise.to(central.dtype))


def make_block_es(args, tok, block_prompts, teacher, device, seeds,
                  permute=False, instruct=True):
    """BLOCK-BATCHED ES-teacher generation. The N ES teachers are FIXED across
    the block; for each teacher we batch-generate every block prompt R times in
    one call, then regroup by prompt. Returns one dict per prompt containing
    N*R rollouts. This supports both the standard N=16, R=1 arm and the
    population-8, two-rollouts-per-member ablation without changing the total
    number of samples used by the group-relative update.

    permute=True (MMLU): each ES teacher m also shows the choices in its OWN random
    order, and the emitted letter is mapped back to the canonical option so the
    stored answer is content, not position (kills the 'always A' mode-vote hack)."""
    n = args.population
    rollouts_per_member = args.rollouts_per_member
    need_ref = args.method == "es_gr_wdpo"   # es_grpo's GRPO loss needs no reference
    # --es-importance additionally needs each rollout's PER-TOKEN logprob under the
    # teacher that generated it, to form the off-policy importance ratio.
    need_old = args.method == "es_grpo" and args.es_importance
    per_prompt = [{"rollouts": [], "answers": [], "ref_logp": [], "old_lp": [],
                   "plen": []} for _ in block_prompts]
    for m in range(n):
        C.perturb(teacher.parameters(), seeds[m], args.sigma, 1.0, device)
        if permute:
            rng = random.Random(seeds[m] * 7919 + 1)     # per-teacher orderings
            built = [C.build_mmlu_rollout_prompt(it, tok, instruct, rng)
                     for it in block_prompts
                     for _ in range(rollouts_per_member)]
            prompts = [p for p, _ in built]
            perms = [pm for _, pm in built]
        else:
            prompts = [it for it in block_prompts
                       for _ in range(rollouts_per_member)]
            perms = [None] * len(prompts)
        rollouts, plens = C.batched_generate(teacher, tok, prompts,
                                             args.temp, args.max_new, device)
        for expanded_index, (seq, pl) in enumerate(zip(rollouts, plens)):
            bi = expanded_index // rollouts_per_member
            if need_ref:
                rlp = C.sequence_logprob(teacher, seq, pl, requires_grad=False)
                per_prompt[bi]["ref_logp"].append(rlp)
            if need_old:
                # captured while THIS teacher's perturbation is still applied
                per_prompt[bi]["old_lp"].append(
                    C.per_token_logp(teacher, seq, pl, requires_grad=False))
            letter = C.extract_answer(tok.decode(seq[0, pl:], skip_special_tokens=True))
            ans = C.canonical_letter(letter, perms[expanded_index]) if permute else letter
            per_prompt[bi]["rollouts"].append(seq)
            per_prompt[bi]["answers"].append(ans)
            per_prompt[bi]["plen"].append(pl)
        C.perturb(teacher.parameters(), seeds[m], args.sigma, -1.0, device)
    return per_prompt


def make_group_sample(args, tok, prompt, student, device, ref_model=None,
                      permute=False, instruct=True, prompt_idx=0):
    """grpo / sft_good / student_dpo: batch the N samples (same student weights) in
    one call. If ref_model is given (student_dpo), also compute each rollout's
    reference log-prob under it (the frozen base = DPO reference).

    MMLU and MedQA training examples are structured dictionaries when choice
    permutation is enabled.  Build one independently permuted text prompt per
    rollout and map every displayed answer letter back to the canonical option.
    The permutation RNG is keyed by the global prompt visit so crash-resume is
    deterministic.
    """
    if permute:
        rng = random.Random(1_000_003 * args.seed + 9_973 * prompt_idx + 17)
        built = [C.build_mmlu_rollout_prompt(prompt, tok, instruct, rng)
                 for _ in range(args.population)]
        prompts = [text for text, _ in built]
        perms = [mapping for _, mapping in built]
    else:
        prompts = [prompt] * args.population
        perms = [None] * args.population

    rollouts, plens = C.batched_generate(student, tok, prompts,
                                         args.temp, args.max_new, device)
    answers = []
    for rollout, prompt_len, mapping in zip(rollouts, plens, perms):
        answer = C.extract_answer(
            tok.decode(rollout[0, prompt_len:], skip_special_tokens=True)
        )
        answers.append(C.canonical_letter(answer, mapping) if mapping else answer)
    ref_logp = []
    if ref_model is not None:
        for rollout, prompt_len in zip(rollouts, plens):
            ref_logp.append(C.sequence_logprob(
                ref_model, rollout, prompt_len, requires_grad=False
            ))
    # Permuting choices can change tokenized prompt lengths, so callers must
    # retain the length belonging to each rollout rather than reusing plens[0].
    return rollouts, answers, plens, ref_logp


def prompt_len_for(prompt_lens, index):
    """Return rollout ``index``'s prompt length; accept legacy scalar callers."""
    if isinstance(prompt_lens, (list, tuple)):
        return int(prompt_lens[index])
    return int(prompt_lens)


def make_group_es_direct(args, tok, prompt, student, device, seeds,
                         permute=False, instruct=True):
    """Direct ES: each member perturbs the selected full/LoRA parameter vector,
    so the N members cannot be batched together; generate one rollout per member.

    permute=True (MMLU): as in make_block_es, each member also sees its OWN random
    choice ordering and the emitted letter is mapped back to the canonical option,
    so the mode vote is over content rather than position."""
    rollouts, answers = [], []
    es_params = direct_es_parameters(student, args.method)
    plen = None
    for m in range(args.population):
        if permute:
            rng = random.Random(seeds[m] * 7919 + 1)
            gen_prompt, perm = C.build_mmlu_rollout_prompt(prompt, tok, instruct, rng)
        else:
            gen_prompt, perm = prompt, None
        C.perturb(es_params, seeds[m], args.sigma, 1.0, device)
        rs, pls = C.batched_generate(student, tok, [gen_prompt], args.temp,
                                     args.max_new, device)
        C.perturb(es_params, seeds[m], args.sigma, -1.0, device)
        rollouts.append(rs[0]); plen = pls[0]
        letter = C.extract_answer(tok.decode(rs[0][0, plen:], skip_special_tokens=True))
        answers.append(C.canonical_letter(letter, perm) if permute else letter)
    return rollouts, answers, plen


def make_group_es_direct_lora_batched(args, tok, prompt, student, device,
                                      seeds, central_params, member_names,
                                      member_parameter_sets, permute=False,
                                      instruct=True):
    """Generate the entire LoRA ES population concurrently on one GPU."""
    load_batched_lora_perturbations(
        central_params, member_parameter_sets, seeds, args.sigma, device
    )

    if permute:
        built = []
        for member, seed in enumerate(seeds):
            rng = random.Random(seed * 7919 + 1)
            built.append(C.build_mmlu_rollout_prompt(
                prompt, tok, instruct, rng
            ))
        prompts = [text for text, _ in built]
        permutations = [mapping for _, mapping in built]
    else:
        prompts = [prompt] * args.population
        permutations = [None] * args.population

    # Use as many simultaneous rows as requested.  The H200 arm sets this to
    # N=16; a smaller value remains available as an emergency memory fallback.
    batch_size = args.es_lora_batch_size or args.population
    batch_size = max(1, min(batch_size, args.population))

    def generate_chunks(size):
        generated_rollouts, generated_lengths = [], []
        for start in range(0, args.population, size):
            stop = min(start + size, args.population)
            generated, lengths = C.batched_generate(
                student, tok, prompts[start:stop], args.temp, args.max_new,
                device, adapter_names=member_names[start:stop],
            )
            generated_rollouts.extend(generated)
            generated_lengths.extend(lengths)
        return generated_rollouts, generated_lengths

    while True:
        try:
            rollouts, prompt_lens = generate_chunks(batch_size)
            break
        except torch.OutOfMemoryError:
            if batch_size == 1:
                raise
            smaller = max(1, batch_size // 2)
            print(f"[es_direct_lora_batch16] batch={batch_size} OOM; "
                  f"retrying with batch={smaller}", flush=True)
            batch_size = smaller
            args.es_lora_batch_size = smaller
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    answers = []
    for rollout, prompt_len, mapping in zip(
            rollouts, prompt_lens, permutations):
        answer = C.extract_answer(
            tok.decode(rollout[0, prompt_len:], skip_special_tokens=True)
        )
        answers.append(C.canonical_letter(answer, mapping) if mapping else answer)
    return rollouts, answers, prompt_lens


def compute_adv(args, answers, gold, prompt_idx):
    """Group-relative advantage under the configured reward.

    reward='vote': majority vote over the group (test-time RL, label-free).
    reward='gold': ground truth (ordinary supervised GRPO).
    --reward-noise P: on a fixed fraction P of prompts the gold target is swapped
    for a random WRONG answer drawn from that group -- a plausible-looking but
    incorrect target. The prompt is still trained on, so update counts are
    identical across noise levels; only the signal changes. The RNG is keyed on
    the prompt index alone, so the corrupted subsets nest as P grows."""
    if args.reward == "vote":
        return C.group_advantage(answers)
    target = gold
    if args.reward_noise > 0.0:
        rng = random.Random(917_000 + prompt_idx)
        if rng.random() < args.reward_noise:
            wrong = [a for a in answers
                     if a is not None and not C.answers_match(a, gold)]
            if wrong:
                target = rng.choice(wrong)
    return C.group_advantage_target(answers, target)


def update(args, method, student, opt, rollouts, answers, adv, a_star,
           ref_logp, seeds, prompt_len, device, kl_ref=None, old_lp=None):
    """Apply the method-specific update. Returns a scalar loss for logging
    (the direct-ES methods return the mean advantage of accepted members)."""
    lora_params = [p for p in student.parameters() if p.requires_grad]

    if method == "sft_good":
        # Mean-per-token NLL over every rollout matching the modal answer.
        # Unlike GRPO, Mode-SFT does not need a non-modal rollout to create a
        # relative advantage, so an unanimous group remains a valid update.
        goods = [k for k, answer in enumerate(answers) if answer == a_star]
        if not goods:
            return None
        # Backpropagate each selected rollout immediately. Summing tensor
        # losses first retains every [tokens, vocabulary] graph concurrently
        # and can exhaust even a 140-GiB H200. Gradient accumulation gives the
        # same gradient as the mean loss with roughly one graph of peak memory.
        opt.zero_grad()
        total = 0.0
        scale = 1.0 / len(goods)
        for k in goods:
            loss_k = -C.sequence_logprob(
                student, rollouts[k], prompt_len_for(prompt_len, k), True, "mean"
            )
            (loss_k * scale).backward()
            total += float(loss_k.detach())
            del loss_k
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        return total * scale

    if method in ("grpo", "es_grpo"):
        # mean-per-token advantage-weighted policy gradient (no length bias).
        # Identical loss for grpo and es_grpo; they differ ONLY in rollout source
        # (grpo: student sampling; es_grpo: ES-perturbed teachers, block-batched).
        # Backward INCREMENTALLY (one rollout at a time) rather than summing all
        # rollouts into a single graph. Each forward materialises [1,T,vocab] logits
        # (~155MB at T=256, vocab 152k) that must be retained for backward; with
        # N=16 and the KL term that is ~32 retained graphs -> 20GB+ and thrashing.
        # Gradients accumulate in .grad, so this is mathematically identical
        # (grad of a sum == sum of grads) at ~1/N the peak memory.
        n = len(adv)
        opt.zero_grad()
        total = 0.0
        for k in range(n):
            if old_lp:
                # OFF-POLICY CORRECTION: rollout k came from teacher_k, not from the
                # student, so reweight by the PPO-clipped per-token importance ratio
                # pi_theta(y_k)/pi_teacher_k(y_k) instead of using the raw logprob.
                lp_tok = C.per_token_logp(
                    student, rollouts[k], prompt_len_for(prompt_len, k), True
                )
                ratio = torch.exp(lp_tok - old_lp[k])
                a = float(adv[k])
                surr = torch.minimum(
                    ratio * a,
                    torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * a)
                l = -surr.mean() / n
            else:
                rollout_prompt_len = prompt_len_for(prompt_len, k)
                lp = C.sequence_logprob(
                    student, rollouts[k], rollout_prompt_len, True, "mean"
                )
                l = -(adv[k].to(device) * lp) / n
            if args.kl_lambda > 0:                    # GRPO-style KL(pi_theta||pi_0)
                l = l + args.kl_lambda * C.kl_to_base(
                    student, kl_ref, rollouts[k],
                    prompt_len_for(prompt_len, k)) / n
            l.backward()                              # frees this rollout's graph
            total += float(l.detach())
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0); opt.step()
        return total

    if method in ("es_direct", "es_direct_lora", "es_direct_lora_batch16"):
        # Direct ES update over the selected full/LoRA parameter vector:
        # theta += alpha/N * sum_k adv_k * eps_k (no backprop).
        C.es_step(direct_es_parameters(student, method), seeds,
                  [float(a) for a in adv],
                  args.sigma, args.alpha, len(adv), device)
        return float(adv.abs().mean())

    if method in ("es_gr_wdpo", "student_dpo"):
        # weighted DPO (+ optional SFT term, RPO). es_gr_wdpo: references are the
        # per-rollout ES teachers. student_dpo: references are all frozen base pi_0.
        pairs = C.select_pairs(adv, answers, a_star, args.tau, args.max_pairs)
        if not pairs:
            return None
        loss = torch.zeros((), device=device)
        for (i, j) in pairs:
            w = float(min(max(adv[i] - adv[j], 0.0), args.w_max))
            pos_prompt_len = prompt_len_for(prompt_len, i)
            neg_prompt_len = prompt_len_for(prompt_len, j)
            pos_sum, pos_n = C.completion_logprob(
                student, rollouts[i], pos_prompt_len, True
            )
            neg_sum, _ = C.completion_logprob(
                student, rollouts[j], neg_prompt_len, True
            )
            s_pos = pos_sum - ref_logp[i]               # sequence-level ratio (DPO)
            s_neg = neg_sum - ref_logp[j]
            loss = loss - w * F.logsigmoid(args.beta * (s_pos - s_neg))
            if args.sft_lambda > 0:
                # SFT/NLL on the CHOSEN response (mean-per-token) -> directly raises
                # the good answer's probability, fixing DPO likelihood displacement
                loss = loss - args.sft_lambda * (pos_sum / pos_n)
            if args.kl_lambda > 0:                    # GRPO-style KL(pi_theta||pi_0)
                loss = loss + args.kl_lambda * (
                    C.kl_to_base(student, kl_ref, rollouts[i], pos_prompt_len)
                    + C.kl_to_base(student, kl_ref, rollouts[j], neg_prompt_len)) / 2
        loss = loss / len(pairs)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0); opt.step()
        return loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["sft_good", "grpo", "grpo_real", "echo", "es_direct",
                             "es_direct_lora", "es_direct_lora_batch16",
                             "es_gr_wdpo", "es_grpo", "student_dpo"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--dataset", default="math500",
                    choices=["math500", "mmlu", "medqa"],
                    help="task to train on (label-free, mode-vote pseudo-reward)")
    ap.add_argument("--permute-choices", action="store_true",
                    help="MMLU/MedQA: show each rollout a random option order and "
                         "vote over canonical CONTENT (kills the always-A hack)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-size", type=int, default=100)
    ap.add_argument("--eval-size", type=int, default=200)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--rollouts-per-member", type=int, default=1,
                    help="es_gr_wdpo/es_grpo: stochastic generations produced by "
                         "each perturbed ES teacher for every prompt")
    ap.add_argument("--rollout-prompt-batch-size", type=int, default=1,
                    help="grpo_real/echo: number of prompts from the frozen-policy "
                         "generation block to flatten into one generate call. Each "
                         "prompt still has --population rollouts and is updated "
                         "separately; 1 preserves the legacy execution path.")
    ap.add_argument("--es-lora-batch-size", type=int, default=0,
                    help="es_direct_lora_batch16: simultaneous perturbed adapters "
                         "per generate call (0 = the full population)")
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=5e-4, help="ES-direct lr")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft-lambda", type=float, default=0.0,
                    help="DPO+SFT (RPO): weight of an NLL term on the chosen "
                         "response added to the DPO loss (0 = pure DPO)")
    ap.add_argument("--kl-lambda", type=float, default=0.0,
                    help="GRPO-style KL penalty: lambda * KL(pi_theta||pi_0) on "
                         "rollout tokens (pi_0 = frozen base kl_ref). 0 = off.")
    ap.add_argument("--clip-eps", type=float, default=0.2,
                    help="grpo_real: PPO clip range for the importance ratio")
    ap.add_argument("--echo-alpha", type=float, default=1.0,
                    help="ECHO weight on whitened token entropy")
    ap.add_argument("--echo-beta", type=float, default=1.0,
                    help="ECHO weight on whitened inverse token confidence")
    ap.add_argument("--echo-s-max", type=float, default=2.0,
                    help="ECHO clamp magnitude for the hybrid exploration score")
    ap.add_argument("--echo-mult-coeff", type=float, default=0.1,
                    help="ECHO multiplicative advantage-shaping strength")
    ap.add_argument("--echo-tail-window", type=int, default=16,
                    help="ECHO response-tail length used by adaptive clipping")
    ap.add_argument("--echo-confidence-topk", type=int, default=5,
                    help="ECHO top-k probabilities averaged for local confidence")
    ap.add_argument("--echo-eps-min", type=float, default=0.05,
                    help="ECHO minimum adaptive PPO clipping epsilon")
    ap.add_argument("--echo-eps-max", type=float, default=0.2,
                    help="ECHO maximum adaptive PPO clipping epsilon")
    ap.add_argument("--echo-kappa", type=float, default=4.0,
                    help="ECHO adaptive-clipping sigmoid slope")
    ap.add_argument("--mode-filter", default="all", choices=["all", "correct", "wrong"],
                    help="ORACLE DIAGNOSTIC: restrict updates to prompts whose "
                         "majority vote matches ('correct') or misses ('wrong') the "
                         "gold answer. Uses gold ONLY to select prompts; the reward "
                         "stays majority-vote. Not a deployable method.")
    ap.add_argument("--reward", default="vote",
                    choices=["vote", "gold", "self_certainty", "evol"],
                    help="'vote' = majority-vote pseudo-label (TTRL, the default). "
                         "'gold' = ground truth (supervised GRPO/RLVR). "
                         "'self_certainty' = INTUITOR: KL(U||pi) confidence, no vote "
                         "and no labels. "
                         "'evol' = EVOL-RL: majority anchor + semantic novelty bands.")
    ap.add_argument("--evol-alpha", type=float, default=0.5,
                    help="EVOL-RL novelty: weight on mean-similarity vs max-similarity")
    ap.add_argument("--ent-coef", type=float, default=0.0,
                    help="EVOL-RL token-level entropy bonus coefficient (0 = off)")
    ap.add_argument("--clip-eps-high", type=float, default=None,
                    help="EVOL-RL asymmetric clipping: upper eps (default = --clip-eps)")
    ap.add_argument("--reward-noise", type=float, default=0.0,
                    help="With --reward gold: on this FRACTION of prompts, replace "
                         "the reward target with a randomly chosen WRONG answer from "
                         "the same group. Every prompt is still trained on, so the "
                         "update count is unchanged -- only the signal is corrupted. "
                         "Corrupted subsets are nested across values (same RNG "
                         "stream), so 0.1's subset is contained in 0.2's.")
    ap.add_argument("--es-importance", action="store_true",
                    help="es_grpo: correct the off-policy bias. Rollout k is drawn "
                         "from teacher (theta_anchor + sigma*eps_k), NOT from the "
                         "student, so the vanilla PG estimator is biased. This "
                         "reweights each rollout by the PPO-clipped per-token ratio "
                         "pi_theta(y_k)/pi_teacher_k(y_k) (clip range --clip-eps).")
    ap.add_argument("--mode-mix", type=float, default=None,
                    help="ORACLE DIAGNOSTIC: target FRACTION of updates whose majority "
                         "vote is correct (e.g. 0.8 = 80%% correct / 20%% wrong). Keeps "
                         "every mode-correct prompt and admits mode-wrong ones only "
                         "while they stay under the target share. Simulates a labeller "
                         "of known accuracy; not a deployable method.")
    ap.add_argument("--snap-every", type=int, default=0,
                    help="if >0, snapshot the student to {out}/snapNNNNNN every "
                         "N prompt-visits (for drift/direction measurement)")
    ap.add_argument("--init-adapter", default=None,
                    help="resume: load LoRA weights from this snapshot dir")
    ap.add_argument("--start-visit", type=int, default=0,
                    help="resume: skip the first N prompt-visits (already done by "
                         "the run that produced --init-adapter)")
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--w-max", type=float, default=2.0)
    ap.add_argument("--max-pairs", type=int, default=4)
    ap.add_argument("--p-high", type=float, default=0.95)
    ap.add_argument("--freq-filter", type=float, default=0.0,
                    help="TAIL TRIMMING for ES: keep only this FRACTION of each "
                         "group's rollouts, ranked by how frequent their answer is "
                         "(0.5 = keep the top half). Drops the idiosyncratic "
                         "rollouts produced by extreme perturbations. 0 = off.")
    ap.add_argument("--block-size", type=int, default=10)
    ap.add_argument("--lr", type=float, default=None,
                    help="override; else a per-method standard default is used")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1,
                    help="outer passes over the train prompts (regenerates rollouts)")
    ap.add_argument("--save-every-epoch", type=int, default=0,
                    help="if >0, additionally snapshot the model to {out}/epochNN "
                         "every K epochs (for drift/direction measurement)")
    ap.add_argument("--inner-epochs", type=int, default=1,
                    help="es_gr_wdpo/es_grpo: reuse each block's rollouts this many "
                         "gradient passes before re-anchoring (generate once, train N times)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.population <= 0:
        ap.error("--population must be positive")
    if args.rollouts_per_member <= 0:
        ap.error("--rollouts-per-member must be positive")
    if args.rollout_prompt_batch_size <= 0:
        ap.error("--rollout-prompt-batch-size must be positive")
    if args.echo_confidence_topk <= 0 or args.echo_tail_window <= 0:
        ap.error("ECHO confidence top-k and tail window must be positive")
    if not 0 <= args.echo_eps_min <= args.echo_eps_max:
        ap.error("ECHO clipping requires 0 <= eps-min <= eps-max")
    out = args.out or f"out/{args.method}"

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = dict(float32=torch.float32, float16=torch.float16,
                 bfloat16=torch.bfloat16)[args.dtype]

    # per-method standard LRs (loss scales differ: SFT/GRPO are mean-per-token,
    # DPO is sequence-level summed). Direct ES uses --alpha, not an optimizer.
    DEFAULT_LR = {"sft_good": 5e-5, "grpo": 1e-5, "grpo_real": 1e-5,
                  "echo": 1e-5,
                  "es_gr_wdpo": 5e-6, "es_grpo": 1e-5, "student_dpo": 5e-6}
    lr = args.lr if args.lr is not None else DEFAULT_LR.get(args.method, 1e-5)

    print(f"[{args.method}] loading {args.model} ... (lr={lr})")
    tok, student, teacher, kl_ref = build_models(args, dtype, device)
    direct_es = args.method in (
        "es_direct", "es_direct_lora", "es_direct_lora_batch16"
    )
    # Direct ES never backpropagates. Evaluation mode removes needless training
    # behavior such as dropout while stochasticity still comes from sampling and
    # the explicit ES perturbations.
    student.eval() if direct_es else student.train()
    opt = None if direct_es else torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=lr)

    # Restore the OPTIMIZER along with the weights. Without this, every crash-
    # resume silently reset AdamW's moment estimates, so the first steps after a
    # resume were unnormalised and effectively much larger. Runs that crashed
    # more therefore trained differently -- which made arms that differed only in
    # crash count non-comparable. A resume is now a true continuation.
    if opt is not None and args.init_adapter:
        opt_path = os.path.join(args.init_adapter, "optimizer.pt")
        if os.path.exists(opt_path):
            try:
                opt.load_state_dict(torch.load(opt_path, map_location=device))
                print(f"[{args.method}] restored optimizer state from {opt_path}")
            except Exception as e:              # shape/param-group mismatch
                print(f"[{args.method}] WARNING: optimizer state not restored ({e}); "
                      f"continuing with a fresh optimizer")
        else:
            print(f"[{args.method}] WARNING: no optimizer.pt in {args.init_adapter} "
                  f"(pre-fix snapshot); continuing with a fresh optimizer")

    instruct = "Instruct" in args.model
    permute = args.permute_choices and args.dataset in ("mmlu", "medqa")
    train_prompts, _ = C.load_task(args.dataset, tok, instruct, args.eval_size,
                                   args.train_size, args.seed, permute)
    print(f"[{args.method}] {len(train_prompts)} {args.dataset} train prompts "
          f"(labels hidden{', permuted choices' if permute else ''})")

    train_golds = None
    if args.reward == "gold":
        if args.dataset != "gsm8k":
            raise SystemExit("--reward gold is implemented for gsm8k only")
        train_golds = C.load_gsm8k_train_golds(args.eval_size, args.train_size, args.seed)
        print(f"[{args.method}] SUPERVISED reward: r_k = 1[a_k == gold]"
              + (f", with {args.reward_noise:.0%} of prompts given a WRONG target"
                 if args.reward_noise > 0 else ""))
    if args.mode_filter != "all" or args.mode_mix is not None:
        if args.dataset != "gsm8k":
            raise SystemExit("--mode-filter/--mode-mix are implemented for gsm8k only")
        train_golds = C.load_gsm8k_train_golds(args.eval_size, args.train_size, args.seed)
        if args.mode_mix is not None:
            print(f"[{args.method}] ORACLE mode-mix={args.mode_mix:.2f}: targeting "
                  f"{args.mode_mix*100:.0f}% mode-correct / {(1-args.mode_mix)*100:.0f}% "
                  f"mode-wrong updates (reward itself stays majority-vote)")
        else:
            print(f"[{args.method}] ORACLE mode-filter='{args.mode_filter}': updating only on "
                  f"prompts whose majority vote {'matches' if args.mode_filter=='correct' else 'misses'}"
                  f" gold (reward itself stays majority-vote)")

    # train_golds is also loaded for --reward gold, so the mode-filter machinery
    # must be gated on the filter flags themselves, not merely on golds existing.
    use_mode_filter = args.mode_filter != "all" or args.mode_mix is not None

    n_members = args.population
    n_used = n_skip = n_filtered = 0
    n_mix_correct = n_mix_wrong = 0
    bs = args.block_size
    visits = args.start_visit       # prompt-visits processed (drives --snap-every)
    run_started = time.monotonic()
    if args.start_visit:
        print(f"[{args.method}] resuming at visit {args.start_visit}")

    batched_direct_state = None
    if args.method == "es_direct_lora_batch16":
        batched_direct_state = batched_lora_es_parameters(student, n_members)
        effective_batch = args.es_lora_batch_size or n_members
        print(f"[es_direct_lora_batch16] simultaneous rollouts="
              f"{min(effective_batch, n_members)}/{n_members}")

    def save_student(path):
        if args.method == "es_direct_lora_batch16":
            # Population adapters are temporary perturbations. Save only theta,
            # the central default adapter, so checkpoints remain small and load
            # through the existing continual runner.
            student.save_pretrained(path, selected_adapters=["default"])
        else:
            student.save_pretrained(path)

    def maybe_snap():
        if args.snap_every and visits % args.snap_every == 0:
            d = f"{out}/snap{visits:06d}"
            save_student(d)
            if opt is not None:               # so a resume continues, not restarts
                torch.save(opt.state_dict(), os.path.join(d, "optimizer.pt"))
            print(f"  [snap] {visits} visits -> {d}")

    if args.method in ("es_gr_wdpo", "es_grpo"):
        # Re-anchor and sample N teachers once per block. Each teacher generates
        # R stochastic completions per prompt; the update sees all N*R rollouts.
        # es_gr_wdpo and es_grpo share this ES pipeline; only update() differs.
        print(f"[{args.method}] ES population={n_members}; "
              f"rollouts/member={args.rollouts_per_member}; "
              f"total/group={n_members * args.rollouts_per_member}")
        n_blocks = (len(train_prompts) + bs - 1) // bs
        start_block = args.start_visit // bs      # resume point, in blocks
        for epoch in range(args.epochs):
            for b in range(n_blocks):
                global_block = epoch * n_blocks + b
                if global_block < start_block:
                    continue                      # done by the run we resumed from
                block_prompts = train_prompts[b * bs:(b + 1) * bs]

                if global_block > 0:
                    nt = C.reanchor_teacher(teacher, student)
                    if global_block % 10 == 0:
                        print(f"  --- re-anchor ES on student (block {global_block}, {nt} tensors) ---")

                seeds = [10_000 * global_block + m for m in range(n_members)]
                per_prompt = make_block_es(args, tok, block_prompts, teacher,
                                           device, seeds, permute, instruct)
                expected_group_size = n_members * args.rollouts_per_member
                if any(len(data["rollouts"]) != expected_group_size
                       for data in per_prompt):
                    raise RuntimeError(
                        "ES rollout regrouping produced an unexpected group size"
                    )

                # Prepare each prompt's advantage/gate ONCE (rollouts are fixed for
                # the block). The expensive generation above happened once; now we
                # can reuse these block_size*N*R rollouts for `inner_epochs`
                # gradient passes.
                prepared = []
                for pi, data in enumerate(per_prompt):
                    if args.freq_filter > 0:
                        keep = max(2, int(round(len(data["answers"]) * args.freq_filter)))
                        idx = C.frequency_filter(
                            data["answers"], keep,
                            random.Random(917_653 + 1000 * global_block + pi))
                        plen = data["plen"]
                        data = {"rollouts": [data["rollouts"][i] for i in idx],
                                "answers": [data["answers"][i] for i in idx],
                                "ref_logp": [data["ref_logp"][i] for i in idx]
                                            if data["ref_logp"] else [],
                                "old_lp": [data["old_lp"][i] for i in idx]
                                          if data["old_lp"] else [],
                                "plen": [plen[i] for i in idx]
                                        if isinstance(plen, (list, tuple)) else plen}
                    adv, a_star, p, valid = C.group_advantage(data["answers"])
                    if not valid or p > args.p_high:
                        n_skip += 1
                        continue
                    if use_mode_filter:
                        hit = bool(C.answers_match(a_star, train_golds[b * bs + pi]))
                        if args.mode_mix is not None:
                            # Admit a prompt only while its class is BEHIND quota.
                            # Symmetric, so it throttles whichever class is over-
                            # represented -- targets both above and below the natural
                            # mode-correct rate work (a rule that only throttles the
                            # wrong class silently returns the natural rate whenever
                            # mode_mix < that rate).
                            t = args.mode_mix
                            tot = n_mix_correct + n_mix_wrong + 1
                            behind = (n_mix_correct < t * tot) if hit else \
                                     (n_mix_wrong < (1.0 - t) * tot)
                            if not behind:
                                n_filtered += 1
                                continue
                            if hit:
                                n_mix_correct += 1
                            else:
                                n_mix_wrong += 1
                        elif (args.mode_filter == "correct") != hit:
                            n_filtered += 1
                            continue
                    prepared.append((data, adv, a_star))

                for _inner in range(args.inner_epochs):
                    for data, adv, a_star in prepared:
                        loss = update(args, args.method, student, opt, data["rollouts"],
                                      data["answers"], adv, a_star, data["ref_logp"],
                                      seeds, data["plen"], device, kl_ref,
                                      old_lp=data["old_lp"])
                        if loss is not None:
                            n_used += 1
                if global_block % 10 == 0:
                    extra = ""
                    if use_mode_filter:
                        extra = f" filtered={n_filtered}"
                        if args.mode_mix is not None:
                            tot = n_mix_correct + n_mix_wrong
                            extra += (f" mix={n_mix_correct}/{tot}"
                                      f"={n_mix_correct/tot:.2f}" if tot else "")
                    print(f"  block {global_block}: used={n_used} skip={n_skip}{extra}")

                visits += len(block_prompts)
                maybe_snap()

                # free this block's GPU tensors before the next block to avoid
                # fragmentation buildup over many blocks (seen to crash ~block 70)
                del per_prompt, prepared
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # checkpoint at the end of every outer epoch so a later crash still
            # leaves a recent, evaluatable model behind
            os.makedirs(out, exist_ok=True)
            save_student(out)
            print(f"  [checkpoint] saved after epoch {epoch} -> {out}")
            if args.save_every_epoch and (
                    epoch % args.save_every_epoch == 0 or epoch == args.epochs - 1):
                save_student(f"{out}/epoch{epoch:02d}")

    elif args.method in ("grpo_real", "echo"):
        # ---- TTRL-faithful GRPO / fixed-budget ECHO ----------------------------
        # Generation phase: a block of prompts is generated from the CURRENT
        # student (= pi_old snapshot; per-token logps captured before any update).
        # Update phase: mini-batch of ONE prompt per gradient step (TTRL uses
        # batch 32 / mini-batch 1), with the PPO-clipped importance-ratio
        # surrogate + k3 KL(pi_theta||pi_0). Within a block, updates after the
        # first move pi_theta off pi_old, so ratio != 1 and the clip is active.
        lora_params = [p for p in student.parameters() if p.requires_grad]
        n_blocks = (len(train_prompts) + bs - 1) // bs
        start_block = args.start_visit // bs      # resume point, in blocks
        for epoch in range(args.epochs):
            for b in range(n_blocks):
                global_block = epoch * n_blocks + b
                if global_block < start_block:
                    continue                      # done by the run we resumed from
                block_prompts = train_prompts[b * bs:(b + 1) * bs]

                # ---- generation: pi_old = current student, logps recorded ----
                batch_data = []
                prompt_batch_size = min(
                    args.rollout_prompt_batch_size, len(block_prompts)
                )
                if global_block == start_block:
                    print(
                        f"[rollout] prompts/generate={prompt_batch_size}; "
                        f"sequences/generate<={prompt_batch_size * n_members}",
                        flush=True,
                    )
                for chunk_start in range(0, len(block_prompts), prompt_batch_size):
                    chunk = block_prompts[
                        chunk_start:chunk_start + prompt_batch_size
                    ]
                    flat_prompts, specifications = [], []
                    for offset, prompt in enumerate(chunk):
                        pi = chunk_start + offset
                        if permute:
                            # Every rollout keeps its own deterministic option order.
                            rng = random.Random(9973 * global_block + pi)
                            built = [
                                C.build_mmlu_rollout_prompt(
                                    prompt, tok, instruct, rng
                                )
                                for _ in range(n_members)
                            ]
                            generated_prompts = [p for p, _ in built]
                            perms = [pm for _, pm in built]
                        else:
                            generated_prompts = [prompt] * n_members
                            perms = [None] * n_members
                        begin = len(flat_prompts)
                        flat_prompts.extend(generated_prompts)
                        specifications.append(
                            (begin, begin + n_members, perms)
                        )

                    flat_rollouts, flat_plens = C.batched_generate(
                        student, tok, flat_prompts,
                        args.temp, args.max_new, device,
                    )
                    for begin, end, perms in specifications:
                        rollouts = flat_rollouts[begin:end]
                        plens = flat_plens[begin:end]
                        answers = []
                        for r, pl, pm in zip(rollouts, plens, perms):
                            letter = C.extract_answer(
                                tok.decode(r[0, pl:], skip_special_tokens=True)
                            )
                            answers.append(
                                C.canonical_letter(letter, pm)
                                if permute else letter
                            )
                        if args.method == "echo":
                            policy_stats = [
                                C.per_token_policy_stats(
                                    student, r, pl,
                                    args.echo_confidence_topk,
                                )
                                for r, pl in zip(rollouts, plens)
                            ]
                            old_lp = [stats[0] for stats in policy_stats]
                            entropies = [stats[1] for stats in policy_stats]
                            confidences = [stats[2] for stats in policy_stats]
                        else:
                            with torch.no_grad():
                                old_lp = [
                                    C.per_token_logp(student, r, pl, False)
                                    for r, pl in zip(rollouts, plens)
                                ]
                            entropies, confidences = [], []
                        # Reward extras are captured under the same frozen pi_old.
                        extra = None
                        if args.reward == "self_certainty":
                            extra = [
                                C.self_certainty(student, r, pl)
                                for r, pl in zip(rollouts, plens)
                            ]
                        elif args.reward == "evol":
                            # The frozen base is the stable semantic encoder.
                            extra = C.rollout_embeddings(
                                kl_ref or student, rollouts, plens
                            )
                        batch_data.append(
                            (rollouts, plens, answers, old_lp, extra,
                             entropies, confidences)
                        )

                # ---- update: one prompt per step, clipped ratio + KL ----
                for pi, data in enumerate(batch_data):
                    (rollouts, plens, answers, old_lp, extra,
                     entropies, confidences) = data
                    visits += 1
                    if args.reward == "self_certainty":
                        # INTUITOR: no vote at all -- reward is the model's own confidence
                        adv, p, valid = C.group_advantage_rewards(extra)
                        a_star = None
                    elif args.reward == "evol":
                        # EVOL-RL: majority anchor selects the band, novelty ranks within it
                        _, a_star, p, valid = C.group_advantage(answers)
                        if valid:
                            adv, _, valid = C.group_advantage_rewards(
                                C.evol_rewards(answers, a_star, extra, args.evol_alpha))
                    else:
                        adv, a_star, p, valid = compute_adv(
                            args, answers, train_golds[b * bs + pi] if train_golds else None,
                            b * bs + pi)
                    if not valid or (args.reward in ("vote", "evol") and p > args.p_high):
                        n_skip += 1
                        maybe_snap()
                        continue
                    if use_mode_filter:
                        # same oracle diagnostic as the ES path (see there for the rule)
                        hit = bool(C.answers_match(a_star, train_golds[b * bs + pi]))
                        if args.mode_mix is not None:
                            t = args.mode_mix
                            tot = n_mix_correct + n_mix_wrong + 1
                            behind = (n_mix_correct < t * tot) if hit else \
                                     (n_mix_wrong < (1.0 - t) * tot)
                            if not behind:
                                n_filtered += 1
                                maybe_snap()
                                continue
                            if hit:
                                n_mix_correct += 1
                            else:
                                n_mix_wrong += 1
                        elif (args.mode_filter == "correct") != hit:
                            n_filtered += 1
                            maybe_snap()
                            continue
                    # Backward INCREMENTALLY, one rollout at a time, as in the ES
                    # path. Each per_token_logp call retains a [1,T,|V|] logits
                    # tensor (~155MB at T=256, |V|=152k) until backward; summing
                    # all N into one graph OOMs at N=16 on a 24GB card. Gradients
                    # accumulate in .grad, so grad-of-sum == sum-of-grads and this
                    # is mathematically identical at ~1/N the peak memory.
                    n_roll = len(adv)
                    if args.method == "echo":
                        token_advantages, clip_epsilons = C.echo_hybrid_advantages(
                            adv, entropies, confidences,
                            alpha=args.echo_alpha,
                            beta=args.echo_beta,
                            s_max=args.echo_s_max,
                            mult_coeff=args.echo_mult_coeff,
                            tail_window=args.echo_tail_window,
                            eps_min=args.echo_eps_min,
                            eps_max=args.echo_eps_max,
                            kappa=args.echo_kappa,
                        )
                    else:
                        token_advantages = [float(value) for value in adv]
                        clip_epsilons = [args.clip_eps] * n_roll
                    opt.zero_grad()
                    total = 0.0
                    for k in range(n_roll):
                        lp = C.per_token_logp(student, rollouts[k], plens[k], True)
                        ratio = torch.exp(lp - old_lp[k])
                        a = token_advantages[k]
                        epsilon = clip_epsilons[k]
                        surr = torch.minimum(
                            ratio * a,
                            torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * a)
                        l = -surr.mean() / n_roll
                        if args.kl_lambda > 0:
                            # fused k3 KL: reuse lp (has grad), one extra
                            # no-grad forward under the frozen base
                            with torch.no_grad():
                                lpr = C.per_token_logp(kl_ref, rollouts[k],
                                                       plens[k], False)
                            d = lpr - lp
                            l = l + args.kl_lambda * (
                                torch.exp(d) - d - 1.0).mean() / n_roll
                        l.backward()               # frees this rollout's graph
                        total += float(l.detach())
                    loss = torch.tensor(total)     # scalar, for logging only
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0); opt.step()
                    n_used += 1
                    if visits % 50 == 0:
                        mixinfo = ""
                        if use_mode_filter:
                            mixinfo = f" filt={n_filtered}"
                            tot = n_mix_correct + n_mix_wrong
                            if args.mode_mix is not None and tot:
                                mixinfo += f" mix={n_mix_correct/tot:.2f}"
                        print(f"  [{visits}] p={p:.2f} mode={a_star} "
                              f"loss={loss.item():.4f}{mixinfo}")
                    maybe_snap()

                del batch_data
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            os.makedirs(out, exist_ok=True)
            save_student(out)
            print(f"  [checkpoint] saved after epoch {epoch} -> {out}")

    else:
        # ---- PROMPT loop for grpo / sft_good / student_dpo / direct ES ----
        for epoch in range(args.epochs):
            for local_t, prompt in enumerate(train_prompts):
                global_t = epoch * len(train_prompts) + local_t
                if global_t < args.start_visit:
                    continue
                visits += 1

                if args.method == "es_direct_lora_batch16":
                    seeds = [10_000 * global_t + m for m in range(n_members)]
                    central_params, member_names, member_parameter_sets = \
                        batched_direct_state
                    rollouts, answers, plen = make_group_es_direct_lora_batched(
                        args, tok, prompt, student, device, seeds,
                        central_params, member_names, member_parameter_sets,
                        permute, instruct,
                    )
                    ref_logp = []
                elif args.method in ("es_direct", "es_direct_lora"):
                    seeds = [10_000 * global_t + m for m in range(n_members)]
                    rollouts, answers, plen = make_group_es_direct(
                        args, tok, prompt, student, device, seeds, permute, instruct)
                    ref_logp = []
                elif args.method == "student_dpo":
                    # sample from student; reference log-probs under frozen base (teacher slot)
                    seeds = []
                    rollouts, answers, plen, ref_logp = make_group_sample(
                        args, tok, prompt, student, device, ref_model=teacher,
                        permute=permute, instruct=instruct, prompt_idx=global_t)
                else:  # grpo, sft_good
                    seeds = []
                    rollouts, answers, plen, ref_logp = make_group_sample(
                        args, tok, prompt, student, device,
                        permute=permute, instruct=instruct, prompt_idx=global_t)

                adv, a_star, p, valid = C.group_advantage(answers)
                # GRPO needs both positive and negative members; its advantage is
                # zero for an unanimous group. Mode-SFT is positive-only and can
                # legitimately imitate all rollouts when all 16 agree.
                high_consensus_without_contrast = (
                    args.method != "sft_good" and p > args.p_high
                )
                if not valid or high_consensus_without_contrast:
                    n_skip += 1
                    if global_t % 20 == 0:
                        elapsed = max(time.monotonic() - run_started, 1e-6)
                        rate = (visits - args.start_visit) / elapsed * 3600
                        print(f"  [progress] visit={visits} skip "
                              f"(p={p:.2f}, mode={a_star}) rate={rate:.1f} visits/h",
                              flush=True)
                    maybe_snap()
                    continue

                loss = update(args, args.method, student, opt, rollouts, answers, adv,
                              a_star, ref_logp, seeds, plen, device, kl_ref)
                if loss is None:
                    n_skip += 1
                    maybe_snap()
                    continue
                n_used += 1
                if global_t % 10 == 0:
                    elapsed = max(time.monotonic() - run_started, 1e-6)
                    rate = (visits - args.start_visit) / elapsed * 3600
                    print(f"  [progress] visit={visits} p={p:.2f} mode={a_star} "
                          f"loss/score={loss:.4f} rate={rate:.1f} visits/h",
                          flush=True)
                maybe_snap()

            if args.save_every_epoch and (
                    epoch % args.save_every_epoch == 0 or epoch == args.epochs - 1):
                os.makedirs(out, exist_ok=True)
                save_student(f"{out}/epoch{epoch:02d}")
                print(f"  [checkpoint] saved epoch {epoch} -> {out}/epoch{epoch:02d}")

    os.makedirs(out, exist_ok=True)
    save_student(out)
    print(f"[{args.method}] done. used={n_used} skipped={n_skip}. saved -> {out}")


if __name__ == "__main__":
    main()
