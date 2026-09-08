# Pre-Release Inference Check for Cumulative Secret Disclosure

## The Problem

Large language models often have access to sensitive information during a conversation — think of an HR chatbot that knows an employee's salary, or a medical assistant that knows a patient's diagnosis. The standard privacy safeguard is simple: don't let the model directly say the secret out loud.

But that's not enough. A user can ask a sequence of *individually harmless* questions — "Is it above average?" "Is it closer to option A or B?", "Can you rule out X?" — and each answer narrows down what the secret could be. No single response reveals anything explicitly, but after five or six turns, the secret is effectively deanonymized. This is a **composition attack**: privacy loss that accumulates across a session rather than appearing in any one response.

Existing safeguards mostly address two things: access control (who's allowed to ask) and content filtering (does this response contain an obviously sensitive keyword or pattern). Neither of these tracks how much cumulative uncertainty has been removed about the secret over the course of a conversation.

## The Idea

Before the LLM sends a response, treat it as a hypothesis test: if we release this, how much does it shrink the space of possible values the secret could take — combined with everything already revealed earlier in this session? We express this in bits, using entropy — the same mathematical currency used in information theory and in differential privacy's "privacy budget" concept.

If a single response would narrow things down a lot, block it. But just as importantly — if a bunch of *small* narrowings across the session add up past a running budget, block or generalise the response even though it looks individually safe. That second part is the key contribution: a **cumulative, session-level budget**, not just a per-message filter.

## What the Experiment Does

Since we don't have a production LLM with real internal probes available, the project builds a synthetic stand-in:

- A "secret" is drawn from a small, known set of possible values (a salary band, a medical condition, a city, an ID fragment).
- A simulated multi-turn conversation asks a sequence of probing questions.
- For each response, we can *exactly* compute how many bits of entropy it removes, because the candidate set is small and known — this makes the ground truth for "was this response unsafe" objective and checkable, rather than guessed.

On top of that dataset, five policies are compared:

1. **No check** — release everything (worst case, upper bound on leakage).
2. **Keyword filter** — a simple stand-in for today's typical content-filtering approach.
3. **Inference check, oracle** — a sanity check showing the ceiling if leak estimation were perfect.
4. **Inference check + cumulative budget** — the actual proposal.
5. **Inference check, per-turn only** — an ablation, showing what you lose *without* the cumulative term.

## The Result, in Privacy Terms

The comparison that matters most is #4 vs #5. Adding the cumulative budget catches more real leaks (higher recall) meaningfully and reduces the average information leaked per session — precisely because it catches the "death by a thousand cuts" pattern that a per-turn-only filter misses by design. The keyword filter, representative of many deployed systems, performs worse on both precision and overall leak reduction, showing that pattern-matching on individual outputs is a weak proxy for actual information disclosure.

## Why This Matters for Data Privacy

This reframes LLM privacy protection away from "does this output contain a sensitive-looking string" and toward "how much does this output, in the context of everything already said, change what an adversary could infer." That's a much closer match to how real privacy harms actually happen — not from one blunt admission, but from incremental inference — and it borrows directly from differential privacy's idea of a bounded, trackable privacy budget rather than a binary allow/block decision per message.
