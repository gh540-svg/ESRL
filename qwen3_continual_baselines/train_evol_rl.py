"""EVOL-RL baseline launcher using the UPSTREAM novelty reward.

WHY THIS FILE EXISTS
--------------------
`train.py --reward evol` computes semantic novelty with common.evol_rewards,
which differs from YujunZhou/EVOL-RL in two places:

  1. mean similarity  -- common averages over ALL other rollouts; upstream
     averages only over rollouts sharing the same correctness label
     (semantic_novelty.py:810).
  2. rescaling        -- common min-max rescales novelty across the WHOLE
     group; upstream rescales WITHIN each correctness band. This is the
     DOMINANT difference: a single-member band rescales to 1.0 upstream, so a
     lone minority answer scores exactly -0.5 where common drives it to ~-1.0.

Measured over 200 random 8-rollout groups per majority size: identical on
unanimous groups, ~0.25 mean / 0.50 max |reward difference| on every split,
roughly independent of majority size. See evol_rl_upstream.py for the table.

HOW IT WORKS
------------
train.py resolves the reward as `C.evol_rewards(...)` -- a module attribute
looked up at call time -- so swapping common.evol_rewards before invoking
train.main() reuses train.py's grpo_real loop VERBATIM. Nothing is duplicated,
so this baseline cannot drift from the loop the other methods use.

train.py and common.py are NOT edited on disk. The patch lives in this
process only, so `train.py --reward evol` keeps its old behaviour and every
result already recorded stays reproducible.

Usage (same flags as train.py; --reward evol is implied):
  python train_evol_rl.py --population 8 --epochs 4 --block-size 10 \
      --train-size 500 --eval-size 200 --snap-every 200 --max-new 256 \
      --dtype bfloat16 --kl-lambda 0.001 --clip-eps 0.2 --out out_base/evol_rl
"""
import sys

import common as C
import train
from evol_rl_upstream import evol_rewards_upstream


def _evol_rewards_upstream_shim(answers, a_star, embs, alpha=0.5):
    """Signature-compatible stand-in for common.evol_rewards.

    Upstream hardwires the 0.5*avg + 0.5*max blend, so --evol-alpha has no
    upstream counterpart; refuse rather than silently ignore a non-default.
    """
    if abs(alpha - 0.5) > 1e-9:
        raise SystemExit(
            f"--evol-alpha {alpha} has no upstream equivalent: EVOL-RL fixes the "
            f"blend at 0.5*avg + 0.5*max. Drop the flag, or use "
            f"`train.py --reward evol` for the tunable (non-upstream) variant.")
    return evol_rewards_upstream(answers, a_star, embs)


def main():
    argv = sys.argv[1:]
    if "--reward" in argv:
        i = argv.index("--reward")
        if i + 1 < len(argv) and argv[i + 1] != "evol":
            raise SystemExit(f"train_evol_rl.py is EVOL-RL only; got "
                             f"--reward {argv[i + 1]}. Use train.py for others.")
    else:
        argv = ["--reward", "evol"] + argv
    if "--method" not in argv:
        # the --reward dispatch lives in train.py's grpo_real branch
        argv = ["--method", "grpo_real"] + argv

    C.evol_rewards = _evol_rewards_upstream_shim          # process-local patch
    print("[evol_rl] using UPSTREAM intra-group novelty "
          "(evol_rl_upstream.evol_rewards_upstream); common.py untouched on disk",
          flush=True)

    sys.argv = [sys.argv[0]] + argv
    train.main()


if __name__ == "__main__":
    main()
