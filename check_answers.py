"""
check_answers.py - automatic consistency check of Llama's answers (step 2).


For every answer it checks:
  A. the action and urgency are allowed by the safety filter
  B. a critical collision risk got a safe action and High urgency
  C. a lane change was only chosen when that side lane has space
  D. "merge" was only chosen on the on-ramp
  E. the Reason / Safety_Consideration sentences do not contradict the scenario
     (vehicle behind "closing in" when it is not, vehicle ahead "slowing" / "speeding up" when it is not,
      AV "approaching" the vehicle ahead when the gap is growing, wrong collision-risk word,
      a side lane called free when it has no space, a "middle lane" that is not in the data)
Writes check_report_<results file name>.txt. A flag means "look at this one", the text rules are simple on purpose.
"""
import json
import re
import sys

file_arg = sys.argv[1] if len(sys.argv) > 1 else "ngsim_results_20.jsonl"
sys.argv = [sys.argv[0]]                       # so that importing the pilot script does not read our arguments
import run_pilot_ngsim_v5 as pilot             # reuses label() so both scripts use exactly the same rules

check_one = pilot.check_one


def main():
    with open(file_arg, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    out, clean, per_rule = [], 0, {}
    for r in recs:
        fl = check_one(r)
        if not fl:
            clean += 1
        for x in fl:
            per_rule[x[0]] = per_rule.get(x[0], 0) + 1
        a = r["answer"] if isinstance(r["answer"], dict) else {}
        out.append(f"{r['Scenario_ID']} ({r['event_type']}): {a.get('AV_Action', '-')} / {a.get('Urgency', '-')}  ->  "
                   + ("OK" if not fl else "FLAGGED"))
        for x in fl:
            out.append("      " + x)
    n = len(recs)
    head = [f"Answers checked: {n}", f"Consistent (no flag): {clean}/{n} = {100 * clean / max(n, 1):.0f}%",
            "Flags by rule: " + (", ".join(f"{k}: {v}" for k, v in sorted(per_rule.items())) or "none"),
            "A = filter, B = critical risk, C = lane change, D = merge, E = text vs data", ""]
    text = "\n".join(head + out)
    report_name = "check_report_" + re.sub(r"\W+", "_", file_arg.rsplit(".", 1)[0]) + ".txt"
    with open(report_name, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print("\nSaved:", report_name)


if __name__ == "__main__":
    main()
