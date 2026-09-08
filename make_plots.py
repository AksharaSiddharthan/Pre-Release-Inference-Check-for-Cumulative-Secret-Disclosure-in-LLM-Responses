import csv
import matplotlib.pyplot as plt
import numpy as np

with open("results_summary.csv") as f:
    rows = list(csv.DictReader(f))

labels = [r["policy"] for r in rows]
short = ["No check", "Keyword\nfilter", "Inference\n(oracle)", "Inference+\ncumulative\n(proposed)",
         "Inference\nper-turn only\n(ablation)", "Inference+\ncumulative\n(noisy)"]
precision = [float(r["precision"]) for r in rows]
recall = [float(r["recall"]) for r in rows]
f1 = [float(r["f1"]) for r in rows]
fbr = [float(r["false_block_rate"]) for r in rows]
leak = [float(r["avg_cumulative_leak_bits"]) for r in rows]
deanon = [float(r["pct_fully_deanonymized"]) * 100 for r in rows]

colors = ["#999999", "#e07b39", "#7fb37f", "#2b6cb0", "#63b3ed", "#805ad5"]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# 1. Precision/Recall/F1 grouped bar
ax = axes[0, 0]
x = np.arange(len(short))
w = 0.25
ax.bar(x - w, precision, w, label="Precision", color="#2b6cb0")
ax.bar(x, recall, w, label="Recall", color="#e07b39")
ax.bar(x + w, f1, w, label="F1", color="#7fb37f")
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Detection quality: Precision / Recall / F1")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# 2. Avg cumulative leak per session
ax = axes[0, 1]
bars = ax.bar(short, leak, color=colors)
ax.set_ylabel("Avg leak per session (bits)")
ax.set_title("Cumulative information leakage per session")
ax.grid(axis="y", alpha=0.3)
for b, v in zip(bars, leak):
    ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)

# 3. False block rate vs Recall tradeoff (utility vs privacy)
ax = axes[1, 0]
ax.scatter(fbr, recall, s=120, c=colors, zorder=3)
for i, lbl in enumerate(short):
    ax.annotate(lbl.replace("\n", " "), (fbr[i], recall[i]), fontsize=7,
                xytext=(5, 5), textcoords="offset points")
ax.set_xlabel("False block rate (safe responses wrongly blocked)")
ax.set_ylabel("Recall (unsafe responses correctly blocked)")
ax.set_title("Privacy/utility tradeoff")
ax.grid(alpha=0.3)
ax.set_xlim(-0.05, 0.55)
ax.set_ylim(-0.05, 1.1)

# 4. % sessions fully deanonymized
ax = axes[1, 1]
bars = ax.bar(short, deanon, color=colors)
ax.set_ylabel("% sessions fully deanonymized")
ax.set_title("Sessions where secret is fully pinned down")
ax.grid(axis="y", alpha=0.3)
for b, v in zip(bars, deanon):
    ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("results_comparison.png", dpi=150)
print("saved results_comparison.png")

# --- Second figure: per-turn leak trajectory example sessions ---
with open("dataset.csv") as f:
    data = list(csv.DictReader(f))

sessions = {}
for r in data:
    sessions.setdefault(r["session_id"], []).append(r)

# pick 3 example sessions with >=6 turns for a nice trajectory plot
candidates = [sid for sid, rs in sessions.items() if len(rs) >= 6]
picked = candidates[:3]

fig2, ax2 = plt.subplots(figsize=(9, 5.5))
for sid in picked:
    rs = sorted(sessions[sid], key=lambda r: int(r["turn_id"]))
    cum = np.cumsum([float(r["leak_bits"]) for r in rs])
    turns = [int(r["turn_id"]) for r in rs]
    dom = rs[0]["secret_domain"]
    ax2.plot(turns, cum, marker="o", label=f"Session {sid} ({dom})")

ax2.axhline(1.5, color="red", linestyle="--", alpha=0.6, label="Proposed session budget (1.5 bits)")
ax2.set_xlabel("Turn")
ax2.set_ylabel("Cumulative leak (bits)")
ax2.set_title("Example sessions: cumulative information leak grows turn-by-turn\n(if nothing were blocked)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cumulative_leak_trajectories.png", dpi=150)
print("saved cumulative_leak_trajectories.png")
