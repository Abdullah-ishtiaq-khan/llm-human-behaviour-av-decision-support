"""
make_evaluation_charts.py - builds the evaluation charts and numbers for the report, from the project's own files.

Run in the project folder (the one that has run_pilot_ngsim_v5.py and the results/ folder):
    python make_evaluation_charts.py

Needs: matplotlib and numpy   (pip install matplotlib numpy)

It reads
    results/baseline_v5_results_300.jsonl          the 300 answers of the final run
    results/ngsim_review_20_reviewed.txt           the 20 answers that a person scored by hand
    results/gemini_batch0_calibration_20.csv       Gemini's scores for the same 20 answers      (optional)
    results/gemini_batch1.csv, ..._batch2.csv, ..._batch3.csv   Gemini's scores for the 300 answers (optional)
(a Gemini file is simply the CSV lines that Gemini wrote, saved as a text file with the .csv ending)

It writes
    results/figures/E1 ... E7 *.png     the charts (200 dpi, ready for Word)
    results/evaluation_summary.txt      every number used in the charts, for the text of the report
Charts whose input file is missing are skipped, so you can run it again after each Gemini batch.

Three evaluations
    1. Human review      - 20 answers scored by a person (sensible / safe / uses only the given facts)
    2. Real drivers      - what the real NGSIM drivers did in the same situation (a sanity check, not an answer key)
    3. Gemini review     - a second, independent judge that sees only the situation and the answer
"""
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")                      # draw into files, no window needed
import matplotlib.pyplot as plt
import numpy as np

RES = "results"
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
SUMMARY = []                               # lines of evaluation_summary.txt

BLUE, ORANGE, GREY, GREEN, RED = "#2F5597", "#E69F00", "#8C8C8C", "#3E8E5A", "#C0504D"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11.5, "axes.titleweight": "bold", "figure.dpi": 100})


def say(line=""):
    print(line)
    SUMMARY.append(line)


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    say(f"  [chart saved: {path}]")


def wilson(k, n, z=1.96):
    """95% confidence interval for a share k/n (good for small samples)"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def kappa(pairs):
    """Cohen's kappa for two lists of Y/N labels. Near 0 = no better than chance, 0.6+ = good."""
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a == "Y" for a, _ in pairs) / n
    pb = sum(b == "Y" for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


# ------------------------------------------------------------------ load the 300 results
path300 = os.path.join(RES, "baseline_v5_results_300.jsonl")
if not os.path.exists(path300):
    sys.exit("Cannot find " + path300 + ". Run this script in the project folder.")
R = [json.loads(l) for l in open(path300, encoding="utf-8") if l.strip()]
BY_ID = {r["Scenario_ID"]: r for r in R}
say("EVALUATION SUMMARY (made by make_evaluation_charts.py)")
say(f"Results file: {path300}  ({len(R)} scenarios)")

SHORT = {
    "Maintain speed and lane": "Maintain",
    "Reduce speed slightly": "Reduce slightly",
    "Reduce speed and increase following distance": "Reduce + distance",
    "Brake firmly": "Brake firmly",
    "Accelerate gently": "Accelerate gently",
    "Accelerate and merge into the main lanes": "Merge",
    "Prepare to change lane to the left": "Lane left",
    "Prepare to change lane to the right": "Lane right",
}
ORDER = ["Brake firmly", "Reduce + distance", "Reduce slightly", "Maintain", "Accelerate gently", "Merge", "Lane left", "Lane right"]
TYPES = ["close_following", "leader_braking", "new_leader_close", "lane_change", "on_ramp", "normal"]
TYPE_NAME = {"close_following": "Close following", "leader_braking": "Leader braking", "new_leader_close": "New leader close",
             "lane_change": "Lane change", "on_ramp": "On-ramp", "normal": "Normal"}


def act(r):
    return SHORT[r["answer"]["AV_Action"]]


def group(a):
    if a in ("Brake firmly", "Reduce + distance", "Reduce slightly"):
        return "Slow down"
    if a == "Maintain":
        return "Maintain"
    if a in ("Accelerate gently", "Merge"):
        return "Speed up"
    return "Prepare lane change"


def real_class(r):
    d = r["real_driver_next_3s"]["Speed_Change_mps"]
    return "Slowed down" if d <= -0.5 else "Sped up" if d >= 0.5 else "About steady"


# ================================================================== EVALUATION 2: Llama vs real drivers
say("\n=== EVALUATION 2: comparison with what the real drivers did ===")
say("The real driver is NOT an answer key (humans brake late, tailgate, change lane in small gaps).")
say("It is a sanity check: do Llama's recommendations point the same way as real behaviour on average?")

# E1 heat map
G = ["Slow down", "Maintain", "Speed up", "Prepare lane change"]
C = ["Slowed down", "About steady", "Sped up"]
M = np.zeros((len(G), len(C)), int)
for r in R:
    M[G.index(group(act(r))), C.index(real_class(r))] += 1
say("\nTable: Llama's recommendation (rows) against the real driver's speed change in the next 3 s (columns; +-0.5 m/s = steady)")
say("%-22s %12s %13s %9s %8s" % ("", C[0], C[1], C[2], "total"))
for i, g in enumerate(G):
    say("%-22s %12d %13d %9d %8d" % (g, *M[i], M[i].sum()))
fig, ax = plt.subplots(figsize=(6.4, 3.6))
share = M / M.sum(axis=1, keepdims=True)
im = ax.imshow(share, cmap="Blues", vmin=0, vmax=0.8, aspect="auto")
ax.set_xticks(range(3), C)
ax.set_yticks(range(4), [f"{g}\n(n={M[i].sum()})" for i, g in enumerate(G)])
for i in range(4):
    for j in range(3):
        ax.text(j, i, f"{M[i, j]}\n{share[i, j]:.0%}", ha="center", va="center",
                color="white" if share[i, j] > 0.45 else "black", fontsize=9)
ax.set_xlabel("What the real driver did in the next 3 seconds")
ax.set_ylabel("What Llama recommended")
ax.set_title("Llama's recommendation vs real driver behaviour")
ax.spines[:].set_visible(False)
fig.colorbar(im, ax=ax, label="share of the row")
save(fig, "E1_llama_vs_real_driver_heatmap.png")

# E2 mean speed change per Llama action
means, errs, ns = [], [], []
for a in ORDER:
    v = [r["real_driver_next_3s"]["Speed_Change_mps"] for r in R if act(r) == a]
    ns.append(len(v))
    means.append(np.mean(v) if v else 0)
    errs.append(1.96 * np.std(v, ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0)
say("\nAverage real-driver speed change (m/s) in the next 3 s, by Llama's action (n, mean, 95% interval)")
for a, n, m, e in zip(ORDER, ns, means, errs):
    say(f"  {a:<20} n={n:<4} mean={m:+.2f}  (+-{e:.2f})")
fig, ax = plt.subplots(figsize=(6.6, 3.7))
cols = [RED if m < 0 else BLUE for m in means]
ax.barh(range(len(ORDER)), means, xerr=errs, color=cols, ecolor=GREY, capsize=3)
ax.set_yticks(range(len(ORDER)), [f"{a} (n={n}):  {m:+.2f}" for a, n, m in zip(ORDER, ns, means)])
ax.invert_yaxis()
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Average speed change of the real drivers in the next 3 s (m/s)")
ax.set_title("Real drivers' average speed change, by Llama's action")
save(fig, "E2_real_driver_speed_change_by_action.png")

# E3 critical-risk scenarios
sys.argv = [sys.argv[0], "dry"]            # so that importing the pipeline does not start a run
try:
    import run_pilot_ngsim_v5 as pilot
except Exception as e:                     # pragma: no cover
    pilot = None
    say("  (could not import run_pilot_ngsim_v5.py: " + str(e) + ")")
if pilot:
    crit = []
    for r in R:
        a = pilot.label(r["input_to_llama"]).get("Vehicle_Ahead")
        if isinstance(a, dict) and a["Collision_Risk"] == "critical":
            crit.append(r)
    ll = Counter(group(act(r)) for r in crit)
    rl = Counter(real_class(r) for r in crit)
    say(f"\nScenarios with CRITICAL collision risk (time to collision under 4 s): {len(crit)}")
    say(f"  Llama recommended a slow-down in {ll['Slow down']} of {len(crit)}")
    say(f"  The real drivers slowed down in {rl['Slowed down']} of {len(crit)}, stayed steady in {rl['About steady']}, sped up in {rl['Sped up']}")
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    labels = ["Slowed down", "About steady", "Sped up"]
    llv = [ll["Slow down"], ll["Maintain"], ll["Speed up"] + ll["Prepare lane change"]]
    rlv = [rl[k] for k in labels]
    x = np.arange(3)
    ax.bar(x - 0.2, llv, 0.4, color=BLUE, label="Llama (slow down / maintain / speed up or lane)")
    ax.bar(x + 0.2, rlv, 0.4, color=ORANGE, label="Real drivers (slowed / steady / sped up)")
    for i in range(3):
        ax.text(i - 0.2, llv[i] + 0.3, llv[i], ha="center", fontsize=9)
        ax.text(i + 0.2, rlv[i] + 0.3, rlv[i], ha="center", fontsize=9)
    ax.set_xticks(x, ["Slow down", "Steady / maintain", "Speed up / lane"])
    ax.set_ylabel("Number of scenarios")
    ax.set_title(f"The {len(crit)} critical-risk scenarios: Llama vs real drivers")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "E3_critical_risk_llama_vs_real.png")

# lane changes
lane_allowed = sum(any(x.startswith("Prepare to change lane") for x in r["allowed_actions"]) for r in R)
lane_chosen = sum(group(act(r)) == "Prepare lane change" for r in R)
lane_real = sum(bool(r["real_driver_next_3s"]["Changed_Lane"]) for r in R)
say(f"\nLane changes: allowed by the filter in {lane_allowed} scenarios, chosen by Llama in {lane_chosen}, made by the real drivers in {lane_real} of {len(R)}")
say("  (the real drivers were picked partly because many of them changed lane, so this number is high by design)")

# E6 actions by event type
fig, ax = plt.subplots(figsize=(7.2, 3.9))
palette = {"Brake firmly": "#C0504D", "Reduce + distance": "#E07B39", "Reduce slightly": "#E8B04A", "Maintain": "#8C8C8C",
           "Accelerate gently": "#6FA8DC", "Merge": "#2F5597", "Lane left": "#7B5EA7", "Lane right": "#A98BD1"}
left = np.zeros(len(TYPES))
say("\nActions chosen per type of situation")
cnt = {t: Counter(act(r) for r in R if r["event_type"] == t) for t in TYPES}
for a in ORDER:
    vals = np.array([cnt[t][a] for t in TYPES])
    ax.barh(range(len(TYPES)), vals, left=left, color=palette[a], label=a, edgecolor="white", linewidth=0.8)
    left += vals
ax.set_yticks(range(len(TYPES)), [f"{TYPE_NAME[t]} (n={sum(cnt[t].values())})" for t in TYPES])
ax.invert_yaxis()
ax.set_xlabel("Number of scenarios")
ax.set_title("What Llama recommended in each type of situation")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17))
for t in TYPES:
    say(f"  {TYPE_NAME[t]:<18}" + "  ".join(f"{a}={cnt[t][a]}" for a in ORDER if cnt[t][a]))
save(fig, "E6_actions_by_event_type.png")

# ================================================================== automatic checker (consistency)
if pilot:
    say("\n=== Automatic checker (rules A to E), re-run on the saved answers ===")
    first_ok = final_ok = 0
    for r in R:
        r1 = dict(r)
        if r.get("first_answer"):
            r1["answer"] = r["first_answer"]
        if not pilot.check_one(r1):
            first_ok += 1
        if not pilot.check_one(r):
            final_ok += 1
    say(f"Consistent at the first try: {first_ok} of {len(R)} ({first_ok / len(R):.1%})")
    say(f"Consistent after one retry:  {final_ok} of {len(R)} ({final_ok / len(R):.1%})")
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    vals = [100 * first_ok / len(R), 100 * final_ok / len(R)]
    ax.bar(["First answer", "After one retry"], vals, color=[BLUE, GREEN], width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Answers consistent with the data (%)")
    ax.set_title("Automatic check of the explanations")
    save(fig, "E7_checker_first_vs_retry.png")

# ================================================================== EVALUATION 1 + 3: human and Gemini
CRIT = [("sensible", "Sensible"), ("safe", "Safe"), ("facts", "Uses only given facts")]


def read_human():
    p = os.path.join(RES, "ngsim_review_20_reviewed.txt")
    if not os.path.exists(p):
        return {}
    out = {}
    for blk in open(p, encoding="utf-8").read().split("=" * 70):
        m = re.match(r"\s*(N\d{4})\b", blk)
        v = re.search(r"Your review.*?\?\s+([YyNn])\s+([YyNn])\s+([YyNn])\s*:?(.*)", blk)
        if m and v:
            out[m.group(1)] = {"sensible": v.group(1).upper(), "safe": v.group(2).upper(), "facts": v.group(3).upper(),
                               "comment": v.group(4).strip()}
    return out


def read_gemini(path):
    if not os.path.exists(path):
        return {}
    txt = open(path, encoding="utf-8").read()
    pat = re.compile(r"(N\d{4})\s*,\s*([YN])\s*,\s*([YN])\s*,\s*([YN])\s*,\s*(.*?)(?=\s*N\d{4}\s*,\s*[YN]\s*,\s*[YN]|\s*TOTALS|\Z)", re.S)
    return {m.group(1): {"sensible": m.group(2), "safe": m.group(3), "facts": m.group(4), "comment": m.group(5).strip()}
            for m in pat.finditer(txt)}


H = read_human()
G0 = read_gemini(os.path.join(RES, "gemini_batch0_calibration_20.csv"))
say("\n=== EVALUATION 1: human review of 20 answers ===")
if H:
    say(f"Answers scored by the human reviewer: {len(H)}")
    for k, name in CRIT:
        y = sum(v[k] == "Y" for v in H.values())
        lo, hi = wilson(y, len(H))
        say(f"  {name:<24} Yes = {y} of {len(H)} ({y / len(H):.0%}, 95% interval {lo:.0%} to {hi:.0%})")
    all3 = sum(all(v[k] == "Y" for k, _ in CRIT) for v in H.values())
    say(f"  All three Yes            {all3} of {len(H)} ({all3 / len(H):.0%})")
else:
    say("  (results/ngsim_review_20_reviewed.txt not found)")

if H and G0:
    ids = [i for i in H if i in G0]
    say("\n=== EVALUATION 3 (calibration): is Gemini a reliable judge? Gemini vs the human on the same 20 ===")
    say(f"Items scored by both: {len(ids)}")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(3)
    hv = [100 * sum(H[i][k] == "Y" for i in ids) / len(ids) for k, _ in CRIT]
    gv = [100 * sum(G0[i][k] == "Y" for i in ids) / len(ids) for k, _ in CRIT]
    ax.bar(x - 0.2, hv, 0.4, color=BLUE, label="Human reviewer")
    ax.bar(x + 0.2, gv, 0.4, color=ORANGE, label="Gemini")
    for i in range(3):
        ax.text(i - 0.2, hv[i] + 1, f"{hv[i]:.0f}%", ha="center", fontsize=9)
        ax.text(i + 0.2, gv[i] + 1, f"{gv[i]:.0f}%", ha="center", fontsize=9)
    agree_txt = []
    for k, name in CRIT:
        pairs = [(H[i][k], G0[i][k]) for i in ids]
        ag = sum(a == b for a, b in pairs)
        kp = kappa(pairs)
        say(f"  {name:<24} human Yes={sum(a == 'Y' for a, _ in pairs)}  Gemini Yes={sum(b == 'Y' for _, b in pairs)}  "
            f"agreement {ag}/{len(ids)} ({ag / len(ids):.0%})  Cohen's kappa {kp:.2f}")
        agree_txt.append(f"{ag / len(ids):.0%}")
    ax.set_xticks(x, [f"{n}\n(agree {a})" for (_, n), a in zip(CRIT, agree_txt)])
    ax.set_ylim(0, 125)
    ax.set_ylabel("Answers judged Yes (%)")
    ax.set_title(f"Human reviewer vs Gemini on the same {len(ids)} answers")
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=9)
    save(fig, "E4_human_vs_gemini_20.png")
    say("  Disagreements (to be settled by the team by hand):")
    for i in ids:
        d = [name for k, name in CRIT if H[i][k] != G0[i][k]]
        if d:
            say(f"    {i}: {', '.join(d)} | human {H[i]['sensible']}{H[i]['safe']}{H[i]['facts']} | Gemini {G0[i]['sensible']}{G0[i]['safe']}{G0[i]['facts']} | Gemini says: {G0[i]['comment'][:90]}")

G = {}
for k in (1, 2, 3):
    G.update(read_gemini(os.path.join(RES, f"gemini_batch{k}.csv")))
G = {i: v for i, v in G.items() if i in BY_ID}
say("\n=== EVALUATION 3: Gemini's review of the 300 answers ===")
if G:
    say(f"Answers scored by Gemini: {len(G)} of {len(R)}")
    for k, name in CRIT:
        y = sum(v[k] == "Y" for v in G.values())
        lo, hi = wilson(y, len(G))
        say(f"  {name:<24} Yes = {y} of {len(G)} ({y / len(G):.0%}, 95% interval {lo:.0%} to {hi:.0%})")
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    w = 0.26
    x = np.arange(len(TYPES))
    for j, ((k, name), col) in enumerate(zip(CRIT, (BLUE, ORANGE, GREEN))):
        vals = []
        for t in TYPES:
            ids = [i for i in G if BY_ID[i]["event_type"] == t]
            vals.append(100 * sum(G[i][k] == "Y" for i in ids) / len(ids) if ids else 0)
        ax.bar(x + (j - 1) * w, vals, w, color=col, label=name)
    ax.set_xticks(x, [TYPE_NAME[t] for t in TYPES], rotation=15)
    ax.set_ylim(0, 125)
    ax.set_ylabel("Answers judged Yes by Gemini (%)")
    ax.set_title(f"Gemini's review of {len(G)} answers, by type of situation")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center")
    save(fig, "E5_gemini_review_by_event_type.png")
    say("\n  Gemini's Yes-share by type of situation (n, sensible, safe, facts)")
    for t in TYPES:
        ids = [i for i in G if BY_ID[i]["event_type"] == t]
        if ids:
            say(f"    {TYPE_NAME[t]:<18} n={len(ids):<4}" + "  ".join(f"{name.split()[0]}={sum(G[i][k] == 'Y' for i in ids)}/{len(ids)} ({sum(G[i][k] == 'Y' for i in ids) / len(ids):.0%})" for k, name in CRIT))
    # a sheet for the team: every answer where Gemini says "facts = N", so that a person can decide who is right
    bad = [i for i in G if G[i]["facts"] == "N"]
    sheet = os.path.join(RES, "gemini_facts_N_to_check.txt")
    with open(sheet, "w", encoding="utf-8") as f:
        f.write(f"{len(bad)} answers where Gemini says the explanation uses facts that are not in the situation (or are wrong).\n")
        f.write("For each one, read the situation and decide: is Gemini RIGHT? Write Y (Gemini right) or N (Gemini wrong) on the last line.\n")
        for i in sorted(bad):
            r = BY_ID[i]
            sit = pilot.summary(r["input_to_llama"]) if pilot else ""
            a = r["answer"]
            f.write("\n" + "=" * 70 + f"\n{i}  type: {r['event_type']}\nSituation: {sit}\n")
            f.write(f"Llama: {a['AV_Action']} ({a['Urgency']})\nReason: {a['Reason']}\nSafety consideration: {a['Safety_Consideration']}\n")
            f.write(f"Gemini says: {G[i]['comment']}\n\nTeam verdict: is Gemini right?  \n")
    say(f"  Sheet for the team to check Gemini's 'facts = N' cases: {sheet}")
    # does the checker agree with Gemini on the facts criterion?
    if pilot:
        both = [(i, bool(pilot.check_one(BY_ID[i])), G[i]["facts"] == "N") for i in G]
        n_flag = sum(f for _, f, _ in both)
        n_gn = sum(g for _, _, g in both)
        n_both = sum(f and g for _, f, g in both)
        say(f"  Checker flagged {n_flag}; Gemini said 'facts = N' for {n_gn}; both together: {n_both}")
else:
    say("  (no results/gemini_batch1.csv, 2, 3 yet: chart E5 is skipped)")

with open(os.path.join(RES, "evaluation_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(SUMMARY) + "\n")
print("\nSummary written to", os.path.join(RES, "evaluation_summary.txt"))
