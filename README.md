# Pre-Release Inference Check for Secret Disclosure — Synthetic Experiment

## What this demonstrates
A synthetic simulation of an LLM session where a user asks a sequence of
individually-innocuous questions that, combined, let an attacker narrow
down a protected secret (salary band, medical condition, home city, or an
employee-ID fragment). The core idea under test: **before releasing a
response, estimate how many bits of uncertainty about the secret it would
remove (alone and combined with everything already revealed this session);
block/generalize if that exceeds a budget.**

## Files
- `generate_dataset.py` — builds `dataset.csv` (1408 rows / 240 sessions,
  4 secret domains, 4–8 turns/session). Each row has the true secret,
  the candidate-set size before/after a hypothetical release, the implied
  information-theoretic leak in bits, and a ground-truth "is this an unsafe
  release" label.
- `detector.py` — implements and evaluates 5 policies over the dataset and
  writes `results_summary.csv`.
- `dataset.csv`, `results_summary.csv` — generated outputs.

## How to run
```bash
python3 generate_dataset.py   # regenerates dataset.csv (seeded, deterministic)
python3 detector.py           # runs all policies, prints + writes results_summary.csv
```
No dependencies beyond the Python standard library.

## Policies compared
1. **NO_CHECK** — always release (upper bound on leakage).
2. **KEYWORD_FILTER** — existing-style baseline: blocks on suspicious
   keywords/patterns ("the value is", "$", "yes, that's correct", etc.),
   no notion of cumulative disclosure.
3. **INFERENCE_CHECK, oracle, per-turn only** — blocks if a single response's
   leak exceeds a bit threshold. Uses the same ground-truth leak estimate
   used to construct labels, so it is an *oracle upper bound*, not a fair
   comparison to the other rows — included to show the ceiling, not as a
   real result.
4. **INFERENCE_CHECK + CUMULATIVE_BUDGET (proposed)** — looser per-turn
   threshold (so several small narrowings are individually allowed) PLUS a
   running session-wide bits budget, DP-composition style. This is the
   actual proposal: catches attacks that stay under the radar turn-by-turn
   but add up.
5. **Same, per-turn-only ablation** — identical per-turn threshold as (4)
   but *without* the cumulative budget, isolating what the cumulative term
   buys you.
6. **Noisy estimator** — same as (4) but with Gaussian noise (σ=0.3 bits)
   added to the leak estimate, simulating a real probe/attention-based
   estimator instead of an oracle.

## Headline results (240 sessions, 1408 turns)

| Policy | Precision | Recall | F1 | False block rate | Avg leak/session (bits) | % sessions fully deanonymized |
|---|---|---|---|---|---|---|
| No check | – | – | – | 0.000 | 2.877 | 60.4% |
| Keyword filter | 0.204 | 0.500 | 0.290 | 0.459 | 0.920 | 50.0% |
| Inference check (oracle, per-turn) | 1.000 | 1.000 | 1.000 | 0.000 | 0.466 | 54.2% |
| **Inference check + cumulative budget (proposed)** | **0.721** | **0.694** | **0.707** | **0.063** | **0.781** | 55.4% |
| Inference check, per-turn only (ablation) | 1.000 | 0.537 | 0.699 | 0.000 | 0.983 | 55.4% |
| Inference check + cumulative, noisy estimator | 0.671 | 0.761 | 0.713 | 0.088 | 0.706 | 55.4% |

## How to read this honestly
- **Row 3 (oracle) is a sanity check, not a fair result** — its threshold is
  literally the same rule used to generate the ground-truth label, so
  perfect performance is guaranteed by construction. It shows the ceiling
  if leak estimation were perfect; it's not evidence the method works on
  its own.
- **The real comparison is row 4 vs row 5** (proposed vs. per-turn-only
  ablation): adding the cumulative session budget raises recall from
  0.537 → 0.694 and lowers average per-session leakage from 0.983 → 0.781
  bits, at the cost of some false blocks (0% → 6.3%). This is the
  composition-attack argument from the proposal made concrete: per-turn
  checks alone under-catch slow, incremental disclosure.
- **Keyword filtering** (representative of many existing "detect sensitive
  content" systems) has poor precision (0.204) — it blocks lots of safe
  small talk containing "$" or "yes" while still missing half of genuine
  leaks (recall 0.500), and doesn't reduce full-deanonymization risk much
  (50.0% vs 60.4% for no check at all).
- **Noise robustness**: adding realistic estimator noise (σ=0.3 bits, row 6)
  degrades precision (0.721→0.671) and false-block rate (0.063→0.088) but
  *recall actually rises slightly* — noise pushes some borderline leaks
  over threshold too. F1 stays roughly flat (0.707 vs 0.713), suggesting
  the policy is reasonably robust to estimator imperfection at this noise
  level, though this should be stress-tested at higher noise for the
  writeup.


