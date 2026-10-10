"""Shared utilities for the four-method comparison (see SPEC.md §8).

All methods share: rollout generation, the majority-vote pseudo-reward, the
group-relative advantage, and the valid-mode confidence gate. Only the parameter
update differs (implemented in train.py).
"""
import math
import re
from fractions import Fraction
from collections import Counter

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Answer extraction + GSM8K ground-truth parsing
# --------------------------------------------------------------------------- #
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_HASH = re.compile(r"####\s*(-?[\d,\.]+)")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm(s):
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
        # Very long generated digit strings overflow float() to +/-inf.  Keep
        # the original exact string instead of attempting int(inf), which
        # would abort an otherwise valid checkpoint evaluation.
        if not math.isfinite(f):
            return s
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        return s


# The training/eval pipeline is task-agnostic: extract_answer + answers_match
# dispatch on TASK so the SAME mode-vote / advantage / gate code works for
# GSM8K (numeric answers) and MMLU (a letter A-D). Set once via set_task().
TASK = "gsm8k"


def set_task(name):
    global TASK
    if name not in (
        "gsm8k", "mmlu", "arc", "aime", "math500", "medqa",
        "amc23", "aime25", "arc_challenge", "gpqa", "medmcqa",
        "pubmedqa",
    ):
        raise ValueError(f"unknown task {name!r}")
    if name in ("mmlu", "arc", "medqa", "arc_challenge", "gpqa", "medmcqa"):
        TASK = "mmlu"
    elif name == "pubmedqa":
        TASK = "pubmedqa"
    else:
        TASK = "math"


def _extract_number(text: str):
    """Final numeric answer: \\boxed{} > '#### x' > last number."""
    for pat in (_BOXED, _HASH):
        m = pat.findall(text)
        if m:
            return _norm(m[-1])
    m = _NUM.findall(text)
    return _norm(m[-1]) if m else None


_ANS_LETTER = re.compile(r"(?:answer|option|choice)\D{0,8}([A-M])\b", re.I)
_LETTER = re.compile(r"\b([A-M])\b")


def _extract_letter(text: str):
    """Final MMLU choice letter: \\boxed{A} > 'answer is A' > last standalone A-D."""
    for cand in reversed(_BOXED.findall(text)):
        c = cand.strip().upper()
        if c and c[0] in "ABCDEFGHIJKLM":
            return c[0]
    m = _ANS_LETTER.findall(text)
    if m:
        return m[-1].upper()
    m = _LETTER.findall(text.upper())
    return m[-1] if m else None


def _last_boxed(text: str):
    """Return the final ``\\boxed{...}``, including nested LaTeX braces."""
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    depth = 1
    begin = start + len(marker)
    for index in range(begin, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:index]
    return None


def _norm_math(value):
    if value is None:
        return None
    value = str(value).strip().strip("$ ")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("\\,", "").replace("\\!", "").replace(" ", "")
    value = value.replace("^{\\circ}", "").replace("^\\circ", "")
    value = value.rstrip(".。")
    # Canonicalise plain numbers and simple numeric LaTeX fractions.  Symbolic
    # answers stay as normalised LaTeX strings; this is deliberately conservative.
    fraction = re.fullmatch(r"-?\\frac\{(-?\d+)\}\{(-?\d+)\}", value)
    if fraction:
        outer_sign = -1 if value.startswith("-\\frac") else 1
        numerator = int(fraction.group(1))
        denominator = int(fraction.group(2))
        if denominator:
            return str(Fraction(outer_sign * numerator, denominator))
    try:
        number = Fraction(value.replace(",", ""))
        return str(number)
    except (ValueError, ZeroDivisionError):
        return value


def _extract_math(text: str):
    boxed = _last_boxed(text)
    if boxed is not None:
        return _norm_math(boxed)
    hashes = _HASH.findall(text)
    if hashes:
        return _norm_math(hashes[-1])
    final = re.findall(
        r"(?:final\s+answer|answer\s+is)\s*[:=]?\s*([^\n]+)", text, re.I
    )
    if final:
        return _norm_math(final[-1])
    return _extract_number(text)


def _extract_pubmedqa(text: str):
    matches = re.findall(r"\b(yes|no|maybe)\b", text, re.I)
    return matches[-1].lower() if matches else None


def extract_answer(text: str):
    """Task-dispatched final-answer parser."""
    if TASK == "mmlu":
        return _extract_letter(text)
    if TASK == "pubmedqa":
        return _extract_pubmedqa(text)
    return _extract_math(text)


def gsm8k_gold(answer_field: str):
    """Parse the gold answer from a GSM8K 'answer' string (ends with '#### N')."""
    m = _HASH.findall(answer_field)
    return _norm(m[-1]) if m else None


def answers_match(a, b):
    if TASK == "mmlu":
        if a is None or b is None:
            return False
        return str(a).strip().upper() == str(b).strip().upper()
    if TASK == "pubmedqa":
        return a is not None and b is not None and str(a).lower() == str(b).lower()
    a, b = _norm_math(a), _norm_math(b)
    if a is None or b is None:
        return False
    try:
        # Compare as exact rationals.  Converting a generated integer with
        # thousands of digits to float overflows even though Fraction can
        # represent it safely.
        return abs(Fraction(a) - Fraction(b)) < Fraction(1, 1_000_000)
    except (ValueError, ZeroDivisionError):
        return a == b


# --------------------------------------------------------------------------- #
# Mode reward + group-relative advantage + valid-mode gate
# --------------------------------------------------------------------------- #
def group_advantage(answers):
    """Returns (adv, a_star, p, valid_mode). valid_mode requires a genuine unique
    mode: >=2 rollouts agree AND no tie for the top."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, None, 0.0, False
    counts = Counter(valid).most_common()
    a_star, count = counts[0]
    second = counts[1][1] if len(counts) > 1 else 0
    valid_mode = (count >= 2) and (count > second)
    p = count / len(answers)
    rewards = torch.tensor([1.0 if a == a_star else 0.0 for a in answers])
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return adv, a_star, p, valid_mode


def frequency_filter(answers, keep, rng):
    """TAIL TRIMMING: keep the `keep` rollouts whose ANSWERS are most frequent.

    Motivation: with a large ES population, the extreme perturbations in the tail
    of eps ~ N(0,I) produce rollouts that are far off the student's distribution.
    Those rollouts also tend to give idiosyncratic (low-frequency) answers, so
    ranking by answer frequency and dropping the bottom half removes them before
    they contribute a large, badly-scaled gradient.

    Returns the kept INDICES (original order preserved). If no two rollouts agree
    at all -- i.e. every answer is distinct, so frequency carries no information --
    a random subset is returned instead."""
    cnt = Counter(a for a in answers if a is not None)
    if not cnt or max(cnt.values()) == 1:
        idx = list(range(len(answers)))
        rng.shuffle(idx)
        return sorted(idx[:keep])
    freq = [cnt.get(a, 0) for a in answers]        # unparseable answers -> 0
    order = sorted(range(len(answers)), key=lambda i: (-freq[i], i))
    return sorted(order[:keep])


def select_pairs(adv, answers, a_star, tau, max_pairs):
    """Disjoint matching, best-good <-> hardest-bad, each rollout used once."""
    goods = sorted([k for k in range(len(adv)) if adv[k] > 0 and answers[k] == a_star],
                   key=lambda k: adv[k], reverse=True)
    bads = sorted([k for k in range(len(adv)) if adv[k] < 0 and answers[k] != a_star],
                  key=lambda k: adv[k], reverse=True)
    pairs = []
    for i, j in zip(goods, bads):
        if adv[i] - adv[j] > tau:
            pairs.append((i, j))
        if len(pairs) >= max_pairs:
            break
    return pairs


# --------------------------------------------------------------------------- #
# ES perturbation (seed-based, in-place, reversible) over a given param list
# --------------------------------------------------------------------------- #
@torch.no_grad()
def perturb(params, seed, sigma, scale, device):
    """Add scale*sigma*noise(seed) to each tensor in `params`, in place.
    Call again with scale=-1 and the same seed to undo. `params` must be walked
    in the same order both times."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    for p in params:
        noise = torch.randn(p.shape, generator=g, device=device, dtype=torch.float32)
        p.add_((scale * sigma) * noise.to(p.dtype))


@torch.no_grad()
def es_step(params, seeds, advantages, sigma, alpha, n, device):
    """ES-direct parameter update on `params` (theta += alpha/n * sum_k adv_k eps_k).
    Regenerates eps_k from each seed (memory-free)."""
    for s, a in zip(seeds, advantages):
        coeff = float(alpha * a / n)
        if coeff == 0.0:
            continue
        g = torch.Generator(device=device)
        g.manual_seed(int(s))
        for p in params:
            noise = torch.randn(p.shape, generator=g, device=device, dtype=torch.float32)
            p.add_(coeff * noise.to(p.dtype))


# --------------------------------------------------------------------------- #
# Log-prob of a completion
# --------------------------------------------------------------------------- #
def sequence_logprob(model, input_ids, prompt_len, requires_grad=False, reduce="sum"):
    """Log-prob of the completion tokens. reduce='sum' = sequence-level total
    (used by DPO, which is inherently sequence-level); reduce='mean' = per-token
    average (used by SFT and GRPO, so the loss does not scale with length)."""
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        logits = model(input_ids).logits
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = input_ids[:, 1:]
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        comp = tok_logp[:, prompt_len - 1:]
        return comp.mean() if reduce == "mean" else comp.sum()


def per_token_logp(model, input_ids, prompt_len, requires_grad=False):
    """Per-completion-token logprob vector [n_comp] (for the GRPO-style KL term)."""
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        logits = model(input_ids).logits
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = input_ids[:, 1:]
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return tok_logp[:, prompt_len - 1:].squeeze(0)


def per_token_policy_stats(model, input_ids, prompt_len, confidence_topk=5):
    """Return old-policy log-probability, entropy, and ECHO confidence.

    ECHO defines local confidence as the mean probability of the top-k tokens at
    each generation step.  These statistics are detached rollout metadata: the
    later policy update is differentiated only through the new-policy log-prob.
    """
    if confidence_topk <= 0:
        raise ValueError("confidence_topk must be positive")
    with torch.no_grad():
        logits = model(input_ids).logits[:, prompt_len - 1:-1, :].float()
        log_probs = F.log_softmax(logits, dim=-1)
        targets = input_ids[:, prompt_len:]
        selected = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        probabilities = log_probs.exp()
        entropy = -(probabilities * log_probs).sum(dim=-1)
        k = min(int(confidence_topk), probabilities.shape[-1])
        confidence = probabilities.topk(k, dim=-1).values.mean(dim=-1)
    return selected.squeeze(0), entropy.squeeze(0), confidence.squeeze(0)


def echo_hybrid_advantages(base_advantages, entropies, confidences,
                           alpha=1.0, beta=1.0, s_max=2.0,
                           mult_coeff=0.1, tail_window=16,
                           eps_min=0.05, eps_max=0.2, kappa=4.0):
    """Apply ECHO's token-level advantage shaping and adaptive PPO clipping.

    This follows the released ECHO implementation: whiten token entropies and
    inverse top-k confidences over the rollout group, form

        S_t = alpha*z(H_t) + beta*z(1-C_t),
        A_t = A_group * (1 + mult_coeff*S_t),

    then obtain one clipping epsilon per rollout from its tail confidence.
    The official additive coefficient defaults to zero and is therefore not
    included here.  Inputs are variable-length per-rollout tensors.
    """
    if not (len(base_advantages) == len(entropies) == len(confidences)):
        raise ValueError("ECHO advantage/statistic group lengths do not match")
    if not entropies:
        return [], []
    if any(e.numel() != c.numel() for e, c in zip(entropies, confidences)):
        raise ValueError("ECHO entropy/confidence token lengths do not match")

    flat_entropy = torch.cat([value.float() for value in entropies])
    flat_inv_conf = torch.cat([
        1.0 - value.float().clamp(0.0, 1.0) for value in confidences
    ])

    def whiten(values):
        if values.numel() <= 1:
            return torch.zeros_like(values)
        variance = values.var(unbiased=True)
        return (values - values.mean()) * torch.rsqrt(variance + 1e-8)

    score = alpha * whiten(flat_entropy) + beta * whiten(flat_inv_conf)
    if s_max is not None:
        score = score.clamp(-float(s_max), float(s_max))

    shaped, clip_epsilons = [], []
    offset = 0
    for base_advantage, confidence in zip(base_advantages, confidences):
        length = confidence.numel()
        local_score = score[offset:offset + length]
        offset += length
        shaped.append(
            torch.as_tensor(base_advantage, device=local_score.device,
                            dtype=local_score.dtype)
            * (1.0 + float(mult_coeff) * local_score)
        )
        span = length if tail_window <= 0 else min(int(tail_window), length)
        tail_confidence = confidence.float()[-span:].mean().clamp(0.0, 1.0)
        epsilon = float(eps_min) + (float(eps_max) - float(eps_min)) * torch.sigmoid(
            float(kappa) * (1.0 - tail_confidence)
        )
        clip_epsilons.append(epsilon)
    return shaped, clip_epsilons


def kl_to_base(student, ref_model, input_ids, prompt_len):
    """k3 unbiased estimator of KL(pi_theta || pi_0) on a rollout's completion
    tokens. pi_0 = a SEPARATE frozen base model `ref_model` (NOT student's adapter
    disabled — repeatedly toggling the PEFT adapter corrupts named_parameters).
    Differentiable through the student; the reference is detached. Non-negative."""
    lp_theta = per_token_logp(student, input_ids, prompt_len, requires_grad=True)
    lp_ref = per_token_logp(ref_model, input_ids, prompt_len, requires_grad=False)
    delta = lp_ref - lp_theta                      # lp_ref carries no grad
    return (torch.exp(delta) - delta - 1.0).mean()


def completion_logprob(model, input_ids, prompt_len, requires_grad=False):
    """One forward; returns (sum_logp_over_completion, n_completion_tokens) so the
    caller can get BOTH the sequence-level sum (for the DPO ratio) and the
    per-token mean (= sum/n, for an SFT term) without a second forward pass."""
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        logits = model(input_ids).logits
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = input_ids[:, 1:]
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        comp = tok_logp[:, prompt_len - 1:]
        return comp.sum(), comp.shape[1]


# --------------------------------------------------------------------------- #
# Re-anchor teacher on the trained student (for es_gr_wdpo block boundaries)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def reanchor_teacher(teacher, student):
    """Copy the student's EFFECTIVE weights (base + LoRA delta) into the frozen
    teacher, so the next ES round perturbs around the trained parameters.

    Computes the merged weight MANUALLY as  W + (B @ A) * scale  instead of calling
    PEFT's merge_adapter()/unmerge_adapter(). Repeatedly toggling the PEFT adapter
    corrupts the module registry (named_parameters -> "too many values to unpack")
    and has segfaulted several runs; this touches no PEFT state at all."""
    cfg = student.peft_config["default"]
    scale = (cfg.lora_alpha / (cfg.r ** 0.5)) if getattr(cfg, "use_rslora", False) \
        else (cfg.lora_alpha / cfg.r)          # PEFT's LoRA scaling

    tdict = teacher.state_dict()
    sdict = student.state_dict()
    copied = 0
    for k, v in sdict.items():
        if "lora_" in k:
            continue
        nk = k.replace("base_model.model.", "").replace("base_layer.", "")
        if nk not in tdict:
            continue
        w = v
        if k.endswith("base_layer.weight"):
            # fold this layer's LoRA in:  W_eff = W + (B @ A) * scale
            pre = k[: -len("base_layer.weight")]
            a = sdict.get(f"{pre}lora_A.default.weight")
            b = sdict.get(f"{pre}lora_B.default.weight")
            if a is not None and b is not None:
                w = v + (b.to(torch.float32) @ a.to(torch.float32)).to(v.dtype) * scale
        tdict[nk].copy_(w)
        copied += 1
    return copied


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate(model, enc, tok, temp, max_new, greedy=False):
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
    if greedy:
        kw.update(do_sample=False)
    else:
        kw.update(do_sample=True, temperature=temp, top_p=0.95)
    with torch.no_grad():
        return model.generate(**enc, **kw)


@torch.no_grad()
def batched_generate(model, tok, prompts, temp, max_new, device, greedy=False,
                     adapter_names=None):
    """Generate ONE completion per prompt in a single batched call (the dominant
    speedup: a 0.5B underuses the GPU at batch=1; batching N prompts costs ~the
    same wall-clock as 1). Returns (rollouts, prompt_lens) where each rollout is
    a clean [1, prompt_len_i + gen_len_i] LongTensor with left/right padding
    stripped, so the batch=1 `sequence_logprob` can consume it unchanged.

    If pad_token == eos_token, the first generated EOS is also the correct end
    of the completion; otherwise this strips ordinary right-padding tokens.

    ``adapter_names`` optionally selects one PEFT LoRA adapter per batch row.
    This lets batched direct ES evaluate several independently perturbed LoRA
    members in one shared-base forward pass.
    """
    old_side = tok.padding_side
    tok.padding_side = "left"                       # required for batched decoding
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    tok.padding_side = old_side
    input_len = enc.input_ids.shape[1]
    prompt_lens = enc.attention_mask.sum(dim=1).tolist()

    kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
    if greedy:
        kw["do_sample"] = False
    else:
        kw.update(do_sample=True, temperature=temp, top_p=0.95)
    if adapter_names is not None:
        if len(adapter_names) != len(prompts):
            raise ValueError("adapter_names must contain one name per prompt")
        kw["adapter_names"] = list(adapter_names)
    out = model.generate(input_ids=enc.input_ids,
                         attention_mask=enc.attention_mask, **kw)

    pad_id = tok.pad_token_id
    rollouts = []
    for b in range(len(prompts)):
        pl = int(prompt_lens[b])
        seq = out[b, input_len - pl:]              # strip LEFT padding -> [pl + gen_max]
        comp = seq[pl:]
        hits = (comp == pad_id).nonzero()
        if len(hits) > 0:
            seq = seq[: pl + int(hits[0])]         # strip RIGHT padding
        rollouts.append(seq.unsqueeze(0))
    return rollouts, [int(x) for x in prompt_lens]


# --------------------------------------------------------------------------- #
# GSM8K loading with a fixed eval/train split (labels hidden during training)
# --------------------------------------------------------------------------- #
def format_prompt(question, tok, instruct):
    body = question + "\n\nThink step by step and put the final answer in \\boxed{}."
    if instruct:
        return tok.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True,
        )
    return body + "\nAnswer:"


def load_gsm8k(tok, instruct, eval_size, train_size, seed=0):
    """Returns (train_prompts, eval_items). eval_items carry gold answers (used
    ONLY at eval time). train_prompts are label-free."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    import random
    random.Random(seed).shuffle(idx)
    eval_idx = idx[:eval_size]
    train_idx = idx[eval_size:eval_size + train_size]

    train_prompts = [format_prompt(ds[i]["question"], tok, instruct) for i in train_idx]
    eval_items = [
        {"prompt": format_prompt(ds[i]["question"], tok, instruct),
         "gold": gsm8k_gold(ds[i]["answer"])}
        for i in eval_idx
    ]
    return train_prompts, eval_items


@torch.no_grad()
def self_certainty(model, input_ids, prompt_len):
    """INTUITOR's intrinsic reward (Zhao et al.): mean over completion tokens of
    KL(Uniform || pi(.|context)) -- how far the next-token distribution is from
    uniform, i.e. how CONFIDENT the model is. No labels and no majority vote.

        KL(U||p) = -log|V| - (1/|V|) * sum_v log p(v)

    Peaked distributions score high, flat ones low."""
    import math
    logits = model(input_ids).logits[:, :-1, :]        # predicts token t+1
    lp = F.log_softmax(logits.float(), dim=-1)
    comp = lp[:, prompt_len - 1:, :]                   # completion positions only
    if comp.shape[1] == 0:
        return 0.0
    V = comp.shape[-1]
    kl = -math.log(V) - comp.mean(dim=-1)              # [1, n_comp]
    return float(kl.mean())


@torch.no_grad()
def rollout_embeddings(model, rollouts, plens):
    """Mean-pooled last hidden state over each rollout's COMPLETION tokens.
    Used as the semantic representation for EVOL-RL's novelty term (avoids a
    separate sentence-embedding dependency; the frozen base model is a stable
    encoder that does not drift as the student trains)."""
    embs = []
    for r, pl in zip(rollouts, plens):
        h = model(r, output_hidden_states=True).hidden_states[-1][0, pl:, :].float()
        v = h.mean(0) if h.shape[0] else torch.zeros(model.config.hidden_size,
                                                     device=r.device)
        embs.append(v / (v.norm() + 1e-8))
    return torch.stack(embs)


def evol_rewards(answers, a_star, embs, alpha=0.5):
    """EVOL-RL (Majority Drives Selection, Novelty Promotes Variation).

    Selection: the majority answer is the anchor -- majority rollouts land in
    [0.5, 1.0], minority in [-1, -0.5], so correctness dominates novelty.
    Variation: within a band, rank by NOVELTY

        u_i = 1 - (alpha * mean_sim_i + (1 - alpha) * max_sim_i)

    which penalises both conformity to the group and near-duplication of any
    single peer."""
    n = len(answers)
    sim = embs @ embs.T                                # cosine (rows are unit norm)
    sim.fill_diagonal_(-1e9)
    mean_sim = (sim.masked_fill(sim <= -1e8, 0.0).sum(1) / max(n - 1, 1))
    max_sim = sim.max(1).values
    u = 1.0 - (alpha * mean_sim + (1.0 - alpha) * max_sim)
    u = (u - u.min()) / (u.max() - u.min() + 1e-8)     # -> [0,1] within the group
    rewards = torch.tensor(
        [(0.5 + 0.5 * float(u[i])) if answers_match(answers[i], a_star)
         else (-1.0 + 0.5 * float(u[i])) for i in range(n)])
    return rewards


def group_advantage_rewards(rewards):
    """z-score an arbitrary continuous reward vector into a group advantage."""
    r = rewards if torch.is_tensor(rewards) else torch.tensor(rewards)
    valid = float(r.std()) > 1e-6                      # need variance to learn
    adv = (r - r.mean()) / (r.std() + 1e-8)
    return adv, float(r.mean()), valid


def group_advantage_target(answers, target):
    """Group-relative advantage against an EXPLICIT target answer, instead of the
    majority vote. Used for supervised GRPO (target = gold) and for the
    reward-noise diagnostic (target = a deliberately wrong answer).

    valid requires reward variance inside the group: if every rollout matches the
    target, or none does, sigma_r = 0 and the advantage is degenerate."""
    rewards = torch.tensor(
        [1.0 if answers_match(a, target) else 0.0 for a in answers])
    n_hit = float(rewards.sum())
    valid = 0.0 < n_hit < len(answers)
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return adv, target, n_hit / len(answers), valid


def load_gsm8k_train_golds(eval_size, train_size, seed=0):
    """Gold answers for the TRAIN split, aligned index-for-index with the
    train_prompts returned by load_gsm8k. ORACLE ONLY: used solely by the
    --mode-filter diagnostic to select which prompts get an update. Never
    enters the reward, which stays majority-vote / label-free."""
    from datasets import load_dataset
    import random

    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    train_idx = idx[eval_size:eval_size + train_size]
    return [gsm8k_gold(ds[i]["answer"]) for i in train_idx]


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def format_prompt_mmlu(question, choices, tok, instruct):
    opts = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    body = (f"{question}\n\n{opts}\n\n"
            f"Think step by step and put the final answer letter "
            f"(A through {_LETTERS[len(choices) - 1]}) in \\boxed{{}}.")
    if instruct:
        return tok.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True,
        )
    return body + "\nAnswer:"


def format_prompt_mc_eval(question, choices, tok, instruct):
    """Multiple-choice evaluation prompt for next-token letter scoring."""
    opts = "\n".join(f"{_LETTERS[i]}. {choice}" for i, choice in enumerate(choices))
    body = (
        f"{question}\n\n{opts}\n\n"
        f"Answer with only the single letter A through {_LETTERS[len(choices) - 1]}."
    )
    if instruct:
        return tok.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return body + "\nAnswer:"


def build_mmlu_rollout_prompt(item, tok, instruct, rng):
    """Present the choices in a RANDOM order for this rollout. Returns
    (prompt_str, perm) where perm[displayed_position] = canonical_choice_index,
    so an emitted display-letter can be mapped back to the real option."""
    k = len(item["choices"])
    perm = list(range(k))
    rng.shuffle(perm)
    shown = [item["choices"][perm[i]] for i in range(k)]
    return format_prompt_mmlu(item["q"], shown, tok, instruct), perm


def canonical_letter(letter, perm):
    """Map an emitted DISPLAY letter (in this rollout's shuffled layout) back to
    the CANONICAL choice letter, so mode-vote is over option CONTENT not position.
    A constant-position answer ('always A') thus maps to a different canonical
    option each rollout -> no free consensus."""
    if letter is None:
        return None
    li = _LETTERS.find(letter.upper())
    if li < 0 or li >= len(perm):
        return None
    return _LETTERS[perm[li]]


def load_mmlu(tok, instruct, eval_size, train_size, seed=0, permute=False):
    """MMLU ('all' test split) with the SAME label-free train / gold-eval split as
    GSM8K. Generation-based: each rollout produces a completion, extract_answer
    parses the choice letter, mode-vote gives the pseudo-label. gold = letter.
    permute=True -> train items are STRUCTURED {q, choices} so each rollout can be
    shown a different option order (positional-bias / reward-hacking fix)."""
    from datasets import load_dataset
    import random

    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    eval_idx = idx[:eval_size]
    train_idx = idx[eval_size:eval_size + train_size]

    def mk(i):
        return format_prompt_mmlu(ds[i]["question"], ds[i]["choices"], tok, instruct)

    if permute:
        train_prompts = [{"q": ds[i]["question"], "choices": list(ds[i]["choices"])}
                         for i in train_idx]
    else:
        train_prompts = [mk(i) for i in train_idx]
    # eval is always canonical order; gold = canonical letter
    eval_items = [{"prompt": format_prompt_mc_eval(
                       ds[i]["question"], ds[i]["choices"], tok, instruct),
                   "gold": _LETTERS[int(ds[i]["answer"])]}
                  for i in eval_idx]
    return train_prompts, eval_items


def load_math500(tok, instruct, eval_size, train_size, seed=0, permute=False):
    """Leakage-safe MATH-500 split: 100 held out, 400 for adaptation.

    MATH-500 only publishes a 500-row test split.  The deterministic shuffle is
    shared by every method and model.  Gold solutions never enter label-free
    trainers; supervised SFT loads the same indices separately.
    """
    from datasets import load_dataset
    import random

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if eval_size + train_size > len(idx):
        raise ValueError(
            f"MATH-500 has {len(idx)} rows; eval_size={eval_size} + "
            f"train_size={train_size} is too large"
        )
    eval_idx = idx[:eval_size]
    train_idx = idx[eval_size:eval_size + train_size]
    train_prompts = [format_prompt(ds[i]["problem"], tok, instruct) for i in train_idx]
    eval_items = [
        {
            "prompt": format_prompt(ds[i]["problem"], tok, instruct),
            "gold": _norm_math(ds[i]["answer"]),
        }
        for i in eval_idx
    ]
    return train_prompts, eval_items


def _medqa_item(row):
    options = row["options"]
    if isinstance(options, dict):
        labels = [label for label in _LETTERS if str(options.get(label, "")).strip()]
        choices = [str(options[label]).strip() for label in labels]
    else:
        choices = [str(value).strip() for value in options if str(value).strip()]
        labels = list(_LETTERS[:len(choices)])
    answer = str(row["answer"]).strip().upper()
    if answer not in labels:
        return None
    return {"q": row["question"], "choices": choices, "ans": labels.index(answer)}


def load_medqa(tok, instruct, eval_size, train_size, seed=0, permute=False):
    """MedQA-USMLE native train for adaptation and native test for evaluation."""
    from datasets import load_dataset
    import random

    train_ds = load_dataset("mkieffer/MedQA-USMLE", split="train")
    test_ds = load_dataset("mkieffer/MedQA-USMLE", split="test")
    train_rows = [item for row in train_ds if (item := _medqa_item(row)) is not None]
    test_rows = [item for row in test_ds if (item := _medqa_item(row)) is not None]
    train_rng = random.Random(seed)
    eval_rng = random.Random(seed)
    train_rng.shuffle(train_rows)
    eval_rng.shuffle(test_rows)
    train_rows = train_rows[:train_size]
    test_rows = test_rows[:eval_size]
    if len(train_rows) != train_size:
        raise ValueError(f"MedQA supplied only {len(train_rows)} usable training rows")

    def mk(item):
        return format_prompt_mmlu(item["q"], item["choices"], tok, instruct)

    if permute:
        train_prompts = [
            {"q": item["q"], "choices": list(item["choices"])} for item in train_rows
        ]
    else:
        train_prompts = [mk(item) for item in train_rows]
    eval_items = [
        {
            "prompt": format_prompt_mc_eval(
                item["q"], item["choices"], tok, instruct
            ),
            "gold": _LETTERS[item["ans"]],
            "labels": list(_LETTERS[:len(item["choices"])]),
        }
        for item in test_rows
    ]
    return train_prompts, eval_items


def load_arc(tok, instruct, eval_size, train_size, seed=0, permute=False):
    """ARC-Challenge: a SECOND small-closed-answer-space task, structurally like
    MMLU (4-way multiple choice, one letter) but different content (grade-school
    science rather than academic knowledge).

    Used to test whether the MMLU -> free-form-math damage is specific to MMLU or
    is a property of ANY task whose reward is satisfiable without generating the
    reasoning. Shares MMLU's letter parsing and permutation machinery.

    A handful of ARC items have 3 or 5 options; those are dropped so every item is
    a clean 4-way choice, matching MMLU exactly."""
    from datasets import load_dataset
    import random

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for r in ds:
        ch, lab = r["choices"]["text"], r["choices"]["label"]
        if len(ch) != 4 or r["answerKey"] not in lab:
            continue                       # keep the 4-way subset only
        items.append({"q": r["question"], "choices": list(ch),
                      "ans": lab.index(r["answerKey"])})
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    eval_idx = idx[:eval_size]
    train_idx = idx[eval_size:eval_size + train_size]

    def mk(i):
        return format_prompt_mmlu(items[i]["q"], items[i]["choices"], tok, instruct)

    if permute:
        train_prompts = [{"q": items[i]["q"], "choices": items[i]["choices"]}
                         for i in train_idx]
    else:
        train_prompts = [mk(i) for i in train_idx]
    eval_items = [{"prompt": mk(i), "gold": _LETTERS[items[i]["ans"]]}
                  for i in eval_idx]
    return train_prompts, eval_items


def load_aime(tok, instruct, eval_size, train_size, seed=0):
    """AIME 2024 -- 30 competition problems, integer answers.

    Follows the TTRL protocol exactly: there is no train/eval split. The model is
    trained on the unlabeled test set itself and evaluated on those same 30
    problems; gold answers are used ONLY at eval time. eval_size/train_size are
    therefore ignored (the set is 30 either way)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    items = [{"prompt": format_prompt(r["problem"], tok, instruct),
              "gold": _norm(str(r["answer"]))} for r in ds]
    return [it["prompt"] for it in items], items


def load_task(dataset, tok, instruct, eval_size, train_size, seed=0, permute=False):
    """Dispatch loader + set the global answer-parsing mode for this process."""
    set_task(dataset)
    if dataset == "math500":
        return load_math500(tok, instruct, eval_size, train_size, seed, permute)
    if dataset == "medqa":
        return load_medqa(tok, instruct, eval_size, train_size, seed, permute)
    if dataset == "aime":
        return load_aime(tok, instruct, eval_size, train_size, seed)
    if dataset == "mmlu":
        return load_mmlu(tok, instruct, eval_size, train_size, seed, permute)
    if dataset == "arc":
        return load_arc(tok, instruct, eval_size, train_size, seed, permute)
    return load_gsm8k(tok, instruct, eval_size, train_size, seed)
