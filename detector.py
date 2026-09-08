"""
detector.py

Implements and compares three "should we release this response?" policies
on the synthetic dataset produced by generate_dataset.py:

1. NO_CHECK        : baseline, always release (upper bound on leakage,
                      lower bound on refusals).
2. KEYWORD_FILTER   : simple baseline used by many existing systems --
                      flags a response as unsafe if it contains suspicious
                      keywords/patterns (numbers, "yes"/"no" to a guess,
                      explicit values). Represents "detect sensitive
                      content" style existing approaches.
3. INFERENCE_CHECK  : our proposed pre-release check. It doesn't look at
                      keywords; it estimates, from the (prior_n_candidates,
                      posterior_n_candidates) implied by the response, how
                      many bits of information about the secret would be
                      revealed if this response were sent, GIVEN everything
                      already revealed in the session so far (cumulative).
                      If projected cumulative leakage exceeds a budget,
                      the response is blocked/generalized.

Since we don't have a real LLM in the loop, `posterior_n_candidates` in the
dataset stands in for "the belief update a Bayesian attacker would make
after seeing this response" -- i.e. exactly the quantity a real system would
need to *estimate* (e.g. via a probe model or attention-based detector, as
mentioned in the research proposal). Here we treat it as ground truth for
building/evaluating the *decision rule* on top of it, and we also test a
noisy/estimated version to simulate real-world estimation error.

Metrics reported:
- Precision / Recall / F1 for "correctly blocks unsafe responses"
- False block rate (safe responses wrongly blocked) -> utility cost
- Cumulative leak bits per session (with vs without each policy)
- Sessions where secret is fully deanonymized (posterior candidates == 1)
"""

import csv
import math
import random
from collections import defaultdict

random.seed(0)


def load_dataset(path="dataset.csv"):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    for r in rows:
        r["turn_id"] = int(r["turn_id"])
        r["prior_n_candidates"] = int(r["prior_n_candidates"])
        r["posterior_n_candidates"] = int(r["posterior_n_candidates"])
        r["prior_entropy"] = float(r["prior_entropy"])
        r["posterior_entropy_after_release"] = float(r["posterior_entropy_after_release"])
        r["leak_bits"] = float(r["leak_bits"])
        r["narrows_secret"] = int(r["narrows_secret"])
    return rows


def group_by_session(rows):
    sessions = defaultdict(list)
    for r in rows:
        sessions[r["session_id"]].append(r)
    for sid in sessions:
        sessions[sid].sort(key=lambda r: r["turn_id"])
    return sessions


# ---------------------------------------------------------------------
# Policy 1: no check
# ---------------------------------------------------------------------
def policy_no_check(row, session_state):
    return "release"


# ---------------------------------------------------------------------
# Policy 2: keyword filter (simulates existing "detect sensitive content"
# systems -- pattern/keyword based, no cumulative reasoning)
# ---------------------------------------------------------------------
SUSPICIOUS_MARKERS = [
    "the value is", "yes, that's correct", "no, that's not correct",
    "confirm it is not", "$", "correct.",
]

def policy_keyword_filter(row, session_state):
    text = row["response_candidate"].lower()
    for marker in SUSPICIOUS_MARKERS:
        if marker in text:
            return "block"
    return "release"


# ---------------------------------------------------------------------
# Policy 3: our pre-release inference check (cumulative entropy budget)
# ---------------------------------------------------------------------
def make_inference_policy(bits_per_turn_budget=0.75, session_bits_budget=None, noise_std=0.0):
    """
    Returns a policy function closing over hyperparameters.

    bits_per_turn_budget: max bits a single response may narrow the secret
                           by (per-turn threshold, mirrors the dataset's
                           ground-truth threshold but is a TUNABLE system
                           parameter, not assumed known).
    session_bits_budget:  optional cumulative cap across the whole session
                           (differential-privacy-style composition budget).
                           If cumulative leak would exceed this, block even
                           if the single-turn leak is small.
    noise_std:             stddev of Gaussian noise added to the estimated
                           leak_bits, to simulate imperfect real-world
                           estimation (e.g. from an attention-based probe
                           rather than an oracle).
    """
    def policy(row, session_state):
        estimated_leak = row["leak_bits"] + random.gauss(0, noise_std)
        estimated_leak = max(0.0, estimated_leak)

        cumulative_so_far = session_state.get("cumulative_leak_bits", 0.0)
        projected_cumulative = cumulative_so_far + estimated_leak

        if estimated_leak > bits_per_turn_budget:
            return "block"
        if session_bits_budget is not None and projected_cumulative > session_bits_budget:
            return "generalize"  # softer action: reduce specificity rather than full refuse
        return "release"

    return policy


# ---------------------------------------------------------------------
# Simulation loop: apply a policy turn-by-turn per session, tracking
# cumulative leakage and whether the secret gets fully deanonymized.
# ---------------------------------------------------------------------
def run_policy(sessions, policy_fn, track_cumulative=True):
    results = []
    total_cumulative_leak = []
    fully_deanonymized_sessions = 0

    for sid, rows in sessions.items():
        session_state = {"cumulative_leak_bits": 0.0}
        # Track the policy's OWN mitigated belief-state size, not the
        # dataset's unmitigated ground-truth trace (which assumes every
        # turn was released). This is what actually happened to the
        # attacker's uncertainty under this policy.
        current_candidates_n = rows[0]["prior_n_candidates"]
        min_candidates_seen = current_candidates_n

        for row in rows:
            # If the secret is already fully pinned down under THIS policy's
            # mitigated trace, there is nothing left to leak this turn.
            already_deanonymized = (current_candidates_n == 1)

            decision = policy_fn(row, session_state)
            released = decision == "release"

            if already_deanonymized:
                effective_leak = 0.0
                effective_posterior_n = 1
            elif released:
                effective_leak = row["leak_bits"]
                effective_posterior_n = row["posterior_n_candidates"]
            elif decision == "generalize":
                # Generalization leaks roughly 30% of the bits (partial info)
                effective_leak = row["leak_bits"] * 0.3
                effective_posterior_n = max(1, int(row["posterior_n_candidates"] * 1.5))
                effective_posterior_n = min(effective_posterior_n, current_candidates_n)
            else:  # blocked
                effective_leak = 0.0
                effective_posterior_n = current_candidates_n

            session_state["cumulative_leak_bits"] += effective_leak
            current_candidates_n = effective_posterior_n
            min_candidates_seen = min(min_candidates_seen, effective_posterior_n)

            predicted_unsafe = 1 if decision in ("block", "generalize") else 0
            results.append({
                "session_id": sid,
                "turn_id": row["turn_id"],
                "true_label_narrows_secret": row["narrows_secret"],
                "predicted_block": predicted_unsafe,
                "decision": decision,
                "effective_leak_bits": effective_leak,
            })

        total_cumulative_leak.append(session_state["cumulative_leak_bits"])
        if min_candidates_seen == 1:
            fully_deanonymized_sessions += 1

    return results, total_cumulative_leak, fully_deanonymized_sessions


def compute_classification_metrics(results):
    tp = fp = tn = fn = 0
    for r in results:
        y = r["true_label_narrows_secret"]
        yhat = r["predicted_block"]
        if y == 1 and yhat == 1:
            tp += 1
        elif y == 0 and yhat == 1:
            fp += 1
        elif y == 0 and yhat == 0:
            tn += 1
        elif y == 1 and yhat == 0:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(results) if results else 0.0
    false_block_rate = fp / (fp + tn) if (fp + tn) else 0.0  # safe responses wrongly blocked

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": accuracy, "false_block_rate": false_block_rate,
    }


def summarize(name, results, cumulative_leaks, n_deanon, n_sessions):
    m = compute_classification_metrics(results)
    avg_cum_leak = sum(cumulative_leaks) / len(cumulative_leaks)
    print(f"\n=== {name} ===")
    print(f"Precision: {m['precision']:.3f}  Recall: {m['recall']:.3f}  F1: {m['f1']:.3f}  Accuracy: {m['accuracy']:.3f}")
    print(f"False block rate (safe responses wrongly blocked): {m['false_block_rate']:.3f}")
    print(f"Avg cumulative leak per session (bits): {avg_cum_leak:.3f}")
    print(f"Sessions fully deanonymized (secret narrowed to 1 candidate): {n_deanon}/{n_sessions} "
          f"({n_deanon/n_sessions:.1%})")
    return {
        "policy": name,
        **m,
        "avg_cumulative_leak_bits": avg_cum_leak,
        "pct_fully_deanonymized": n_deanon / n_sessions,
    }


def main():
    rows = load_dataset("dataset.csv")
    sessions = group_by_session(rows)
    n_sessions = len(sessions)

    summary_rows = []

    # Baseline 1: no check
    res, leaks, deanon = run_policy(sessions, policy_no_check)
    summary_rows.append(summarize("NO_CHECK (baseline)", res, leaks, deanon, n_sessions))

    # Baseline 2: keyword filter
    res, leaks, deanon = run_policy(sessions, policy_keyword_filter)
    summary_rows.append(summarize("KEYWORD_FILTER (existing-style baseline)", res, leaks, deanon, n_sessions))

    # Proposed: inference check, per-turn budget only (oracle leak estimate)
    policy = make_inference_policy(bits_per_turn_budget=0.75, session_bits_budget=None, noise_std=0.0)
    res, leaks, deanon = run_policy(sessions, policy)
    summary_rows.append(summarize("INFERENCE_CHECK (per-turn budget, oracle)", res, leaks, deanon, n_sessions))

    # Proposed: inference check + cumulative session budget (DP-style composition).
    # A looser per-turn threshold (1.2 bits) lets several "individually small"
    # narrowings pass turn-by-turn -- this is exactly the composition attack
    # a per-turn-only check misses. The session budget (1.5 bits total) is
    # what catches it.
    policy = make_inference_policy(bits_per_turn_budget=1.2, session_bits_budget=1.5, noise_std=0.0)
    res, leaks, deanon = run_policy(sessions, policy)
    summary_rows.append(summarize("INFERENCE_CHECK + CUMULATIVE_BUDGET (proposed, full)", res, leaks, deanon, n_sessions))

    # Same per-turn-only threshold WITHOUT the cumulative budget, for direct
    # ablation against the row above (isolates the value the session budget adds).
    policy = make_inference_policy(bits_per_turn_budget=1.2, session_bits_budget=None, noise_std=0.0)
    res, leaks, deanon = run_policy(sessions, policy)
    summary_rows.append(summarize("INFERENCE_CHECK, per-turn only, loose threshold (ablation)", res, leaks, deanon, n_sessions))

    # Proposed, but with realistic estimation noise (simulating an imperfect
    # attention-based / probe-based leak estimator instead of an oracle)
    policy = make_inference_policy(bits_per_turn_budget=1.2, session_bits_budget=1.5, noise_std=0.3)
    res, leaks, deanon = run_policy(sessions, policy)
    summary_rows.append(summarize("INFERENCE_CHECK + CUMULATIVE_BUDGET (noisy estimator, std=0.3)", res, leaks, deanon, n_sessions))

    # Write comparison table to CSV
    fieldnames = list(summary_rows[0].keys())
    with open("results_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print("\nWrote results_summary.csv")


if __name__ == "__main__":
    main()
