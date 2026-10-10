"""EVOL-RL novelty reward, matching YujunZhou/EVOL-RL exactly.

common.py's evol_rewards() computes BOTH similarities globally. Upstream
(verl/verl/workers/reward_manager/semantic_novelty.py:810) computes them over
DIFFERENT sets:

    group_similarities = [sim[i, j] for j in group_indices]     # INTRA-GROUP
    avg_similarity = np.mean(group_similarities)                # correct-with-
                                                                # correct, or
                                                                # wrong-with-wrong
    all_similarities = [sim[i, j] for j in all_other_indices]   # GLOBAL
    max_similarity = np.max(all_similarities)
    combined_similarity = 0.5 * avg_similarity + 0.5 * max_similarity
    novelty = 1.0 - combined_similarity

avg over the answer's own correctness group, max over everyone.

Upstream ALSO min-max rescales novelty WITHIN each correctness band; common.py
rescales across the whole group. MEASURED divergence vs common.evol_rewards
(alpha=0.5, 200 random 8-rollout groups per row, max |reward difference|):

    majority   mean |diff|   max |diff|
      8/8         0.000        0.000     (unanimous -- identical)
      7/8         0.281        0.500
      6/8         0.281        0.500
      5/8         0.254        0.500
      4/8         0.250        0.500

So the two agree ONLY on unanimous groups (which the p_high gate skips anyway)
and differ by ~0.25 on average, 0.50 worst case, on every split -- roughly
constant in the majority size, NOT small for lopsided groups.

The BAND-WISE RESCALE dominates, not the intra-group average: a single-member
band always rescales to 1.0 upstream, so a lone minority answer scores exactly
-0.5, where common.py's global rescale drives it toward -1.0.

Upstream fallbacks reproduced: a singleton group falls back to the global
average, and a lone answer uses 0.5 for both terms.

This module is NEW and imported by nothing existing -- common.py and train.py
are untouched, so every result already recorded stays reproducible. Use it via
train_evol_rl.py, or pass evol_rewards_upstream into your own loop.
"""
import torch

from common import answers_match


def evol_rewards_upstream(answers, a_star, embs):
    """Upstream semantic-novelty reward. Returns a [n] tensor.

    answers  list of extracted answers (None allowed)
    a_star   majority answer (the correctness anchor)
    embs     [n, d] unit-norm rollout embeddings
    """
    n = len(answers)
    sim = (embs @ embs.T).float()

    is_correct = [answers_match(a, a_star) for a in answers]

    novelty = torch.zeros(n)
    for i in range(n):
        all_other = [j for j in range(n) if j != i]
        # intra-group = same correctness label, excluding self
        group_idx = [j for j in all_other if is_correct[j] == is_correct[i]]

        if group_idx:
            avg_similarity = float(sim[i, group_idx].mean())
        elif all_other:
            avg_similarity = float(sim[i, all_other].mean())   # upstream fallback
        else:
            avg_similarity = 0.5                                # lone answer

        if all_other:
            max_similarity = float(sim[i, all_other].max())
        else:
            max_similarity = 0.5

        combined = 0.5 * avg_similarity + 0.5 * max_similarity
        novelty[i] = 1.0 - combined

    # Upstream rescales WITHIN each correctness band, not across the whole group.
    rewards = torch.zeros(n)
    for correct, lo in ((True, 0.5), (False, -1.0)):
        idx = [i for i in range(n) if is_correct[i] == correct]
        if not idx:
            continue
        vals = novelty[idx]
        lo_v, hi_v = float(vals.min()), float(vals.max())
        for i in idx:
            rescaled = ((float(novelty[i]) - lo_v) / (hi_v - lo_v)
                        if hi_v > lo_v else 1.0)
            rewards[i] = lo + 0.5 * rescaled
    return rewards
