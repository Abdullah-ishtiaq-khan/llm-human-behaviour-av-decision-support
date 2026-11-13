"""
run_pilot_ngsim.py - Llama pilot on the NGSIM scenarios.

Put this file in the same folder as ngsim_scenarios.jsonl, make sure Ollama is running, then:
    python run_pilot_ngsim.py          (20 scenarios)
    python run_pilot_ngsim.py 5        (quick test with 5 scenarios)

It sends each scenario to your LOCAL Llama and saves:
    ngsim_pilot_results.jsonl   full results
    ngsim_pilot_review.txt      easy-to-read version for checking by hand
Only the "Scenario" part is sent to Llama. The "Meta" part (event type, what the real driver
did next) is kept for you, to help when you review the answers.
Uses only Python's standard library - nothing to install.
"""
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

DATA_FILE = "ngsim_scenarios.jsonl"
MODEL = "llama3.1:8b"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
args = [a for a in sys.argv[1:] if a != "dry"]
DRY = "dry" in sys.argv[1:]            # "python run_pilot_ngsim.py 5 dry" = test the script without Llama
N_SCENARIOS = int(args[0]) if args else 20
SEED = 42

ACTIONS = [
    "Maintain speed and lane",
    "Reduce speed slightly",
    "Reduce speed and increase following distance",
    "Brake firmly",
    "Accelerate gently",
    "Accelerate and merge into the main lanes",
    "Prepare to change lane to the left",
    "Prepare to change lane to the right",
]

SYSTEM_PROMPT = """You are the decision-support module of an autonomous vehicle (AV) driving on a highway in dense traffic.
You are given a structured description of what the AV senses at one moment: its own speed, the human-driven vehicle ahead, the human-driven vehicle behind, and the lanes on its left and right. Distances are in metres, speeds in metres per second, times in seconds.

How to read the data:
- Ahead_Trend / Behind_Trend and Gap_Trend are already worked out for you. Use these words; do not recalculate them.
- Following_Gap tells you how close the AV is to the vehicle ahead: "very short" (under 1 s) is tailgating, "short" is 1 to 2 s, "comfortable" is over 2 s.
- Collision_Risk comes from the time-to-collision: "critical" (under 4 s), "elevated" (4 to 8 s), "none" otherwise.
- Space_For_Lane_Change is "yes" only when the lane exists, nobody is alongside and the gaps are large enough.
- Merge_Lane = true means the AV is on the on-ramp and has to join the main lanes.

Choose exactly ONE action from this list:
""" + "\n".join("- " + x for x in ACTIONS) + """

Guidance:
- Safety comes first. Urgency is High when Collision_Risk is "critical", or when the gap ahead is "very short" and the vehicle ahead is slowing down.
- Brake or slow down only when something ahead makes it necessary. If the vehicle ahead is steady or speeding up and the gap is comfortable, maintaining speed is fine.
- Suggest a lane change only if Space_For_Lane_Change is "yes" and staying in the lane is a worse option.
- On the on-ramp with free space, the right action is to match the speed of traffic and merge.
- Base the reason only on the facts given. Do not invent facts. Do not give low-level commands such as brake percentages or steering angles.

Answer ONLY with a JSON object with these keys: AV_Action, Urgency (Low, Medium or High), Reason (one or two sentences), Safety_Consideration (one sentence)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "AV_Action": {"type": "string", "enum": ACTIONS},
        "Urgency": {"type": "string", "enum": ["Low", "Medium", "High"]},
        "Reason": {"type": "string"},
        "Safety_Consideration": {"type": "string"},
    },
    "required": ["AV_Action", "Urgency", "Reason", "Safety_Consideration"],
}


def trend(change):
    if change is None:
        return None
    return "slowing down" if change <= -0.5 else "speeding up" if change >= 0.5 else "steady"


def gap_trend(closing, who):
    """closing > 0 means the two vehicles are getting closer"""
    if closing is None:
        return None
    if closing > 0.3:
        return f"{who} is closing in"
    if closing < -0.3:
        return "gap is growing"
    return "gap about constant"


def headway_label(h):
    if h is None:
        return None
    return "very short" if h < 1.0 else "short" if h < 2.0 else "comfortable"


def risk_label(ttc):
    if ttc is None:
        return "none"
    return "critical" if ttc < 4 else "elevated" if ttc < 8 else "none"


def side_label(lane):
    """lane = side lane dict or a text saying there is no lane"""
    if not isinstance(lane, dict):
        return {"Lane_Exists": False, "Space_For_Lane_Change": "no"}
    a, b = lane["Nearest_Vehicle_Ahead"], lane["Nearest_Vehicle_Behind"]
    ok = (not lane["Vehicle_Alongside"]) and (a is None or a["Gap_m"] >= 10) and (b is None or b["Gap_m"] >= 10)
    out = dict(lane)
    out["Lane_Exists"] = True
    out["Space_For_Lane_Change"] = "yes" if ok else "no"
    return out


def label(s):
    """turn the raw scenario into the version Llama sees: same facts, plus plain-word labels
    (the raw signed numbers are what Llama kept misreading in the first pilot)"""
    a, b = s["Vehicle_Ahead"], s["Vehicle_Behind"]
    out = {
        "Road_Type": s["Road_Type"],
        "AV_Lane": s["AV_Lane"],
        "Merge_Lane": s["AV_Lane"].lower().startswith("on-ramp"),
        "AV_Speed_mps": s["AV_Speed_mps"],
        "AV_Speed_kmh": s["AV_Speed_kmh"],
        "AV_Speed_Change_Last_3s_mps": s["AV_Speed_Change_Last_3s_mps"],
    }
    if a:
        out["Vehicle_Ahead"] = {
            "Vehicle_Type": a["Vehicle_Type"], "Gap_m": a["Gap_m"], "Speed_mps": a["Speed_mps"],
            "Ahead_Trend": trend(a["Speed_Change_Last_1s_mps"]),
            "Gap_Trend": gap_trend(a["Relative_Speed_mps"], "AV"),
            "Time_Headway_s": a["Time_Headway_s"],
            "Following_Gap": headway_label(a["Time_Headway_s"]),
            "Time_To_Collision_s": a["Time_To_Collision_s"],
            "Collision_Risk": risk_label(a["Time_To_Collision_s"]),
        }
    else:
        out["Vehicle_Ahead"] = "No vehicle ahead in this lane"
    if b:
        out["Vehicle_Behind"] = {
            "Vehicle_Type": b["Vehicle_Type"], "Gap_m": b["Gap_m"], "Speed_mps": b["Speed_mps"],
            "Behind_Trend": gap_trend(b["Relative_Speed_mps"], "vehicle behind"),
            "Time_Headway_s": b["Time_Headway_s"],
            "Collision_Risk": risk_label(b["Time_To_Collision_s"]),
        }
    else:
        out["Vehicle_Behind"] = "No vehicle behind in this lane"
    out["Left_Lane"] = side_label(s["Left_Lane"])
    out["Right_Lane"] = side_label(s["Right_Lane"])
    out["Vehicles_Within_100m_All_Lanes"] = s["Vehicles_Within_100m_All_Lanes"]
    return out


# three short worked examples (invented numbers, not taken from the data)
def _ex(av_lane, kmh, ahead, behind, left, right, near):
    ms = round(kmh / 3.6, 1)
    return {"Road_Type": "Highway", "AV_Lane": av_lane, "AV_Vehicle_Type": "Car", "AV_Speed_mps": ms,
            "AV_Speed_kmh": kmh, "AV_Acceleration_mps2": 0.0, "AV_Speed_Change_Last_3s_mps": 0.0,
            "Vehicle_Ahead": ahead, "Vehicle_Behind": behind, "Left_Lane": left, "Right_Lane": right,
            "Vehicles_Within_100m_All_Lanes": near}


def _veh(gap, spd, change=0.0, rel=0.0, thw=2.0, ttc=None):
    return {"Vehicle_Type": "Car", "Gap_m": gap, "Speed_mps": spd, "Speed_Change_Last_1s_mps": change,
            "Relative_Speed_mps": rel, "Time_Headway_s": thw, "Time_To_Collision_s": ttc}


def _lane(alongside, ahead, behind):
    return {"Vehicle_Alongside": alongside,
            "Nearest_Vehicle_Ahead": {"Gap_m": ahead, "Speed_mps": 10.0} if ahead else None,
            "Nearest_Vehicle_Behind": {"Gap_m": behind, "Speed_mps": 10.0} if behind else None}


NO_L, NO_R = "No lane on the left at this position", "No lane on the right at this position"
EXAMPLES = [
    (_ex("Lane 3 of 5 (1 = leftmost)", 72.0, _veh(14.0, 11.0, -2.5, 9.0, 0.7, 1.6),
         _veh(30.0, 20.0, 0, 0.0, 1.5), _lane(True, 20.0, 12.0), _lane(False, 40.0, 25.0), 30),
     {"AV_Action": "Brake firmly", "Urgency": "High",
      "Reason": "The vehicle ahead is slowing down, the following gap is very short and the collision risk is critical.",
      "Safety_Consideration": "Open the gap before the vehicle ahead slows further."}),
    (_ex("On-ramp lane", 45.0, None, None, NO_L, _lane(False, 60.0, 50.0), 20),
     {"AV_Action": "Accelerate and merge into the main lanes", "Urgency": "Medium",
      "Reason": "The AV is on the on-ramp with no vehicle ahead or behind, so it should match the speed of traffic and join the main lanes.",
      "Safety_Consideration": "Check the gap in the main lane before moving over."}),
    (_ex("Lane 2 of 5 (1 = leftmost)", 50.0, _veh(35.0, 14.0, 0.2, -0.1, 2.5),
         _veh(30.0, 14.0, 0, 0.0, 2.1), _lane(False, 50.0, 40.0), _lane(True, 20.0, 15.0), 25),
     {"AV_Action": "Maintain speed and lane", "Urgency": "Low",
      "Reason": "The vehicle ahead is steady with a comfortable gap and the vehicle behind is not closing in.",
      "Safety_Consideration": "Keep watching the gap ahead as traffic is dense."}),
]


def build_messages(scenario):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_in, ex_out in EXAMPLES:
        msgs.append({"role": "user", "content": "Scenario (JSON):\n" + json.dumps(label(ex_in), ensure_ascii=False)})
        msgs.append({"role": "assistant", "content": json.dumps(ex_out, ensure_ascii=False)})
    msgs.append({"role": "user", "content": "Scenario (JSON):\n" + json.dumps(label(scenario), ensure_ascii=False)
                 + "\n\nRespond with the JSON object only."})
    return msgs


def ask_llama(scenario):
    if DRY:
        return json.dumps({"AV_Action": "(dry run)", "Urgency": "Low", "Reason": "-", "Safety_Consideration": "-"})
    body = {
        "model": MODEL,
        "messages": build_messages(scenario),
        "stream": False,
        "format": SCHEMA,
        "options": {"temperature": 0.2, "num_ctx": 6144, "seed": SEED},
    }
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode("utf-8"))["message"]["content"]


def side(name, lane):
    if not isinstance(lane, dict):
        return f"{name}: {lane}"
    bits = []
    if lane["Vehicle_Alongside"]:
        bits.append("vehicle alongside")
    for k, label in (("Nearest_Vehicle_Ahead", "ahead"), ("Nearest_Vehicle_Behind", "behind")):
        v = lane[k]
        bits.append(f"{label} {v['Gap_m']} m" if v else f"nothing {label} within 100 m")
    return f"{name}: " + ", ".join(bits)


def summary(s):
    """short plain-English version of a scenario, for the person reviewing"""
    L = label(s)
    a, b = L["Vehicle_Ahead"], L["Vehicle_Behind"]
    out = [f"AV: {s['AV_Speed_kmh']} km/h in {s['AV_Lane']}."]
    if isinstance(a, dict):
        ttc = f", time-to-collision {a['Time_To_Collision_s']} s" if a["Time_To_Collision_s"] else ""
        out.append(f"Ahead: {a['Gap_m']} m ({a['Following_Gap']} gap, {a['Time_Headway_s']} s), "
                   f"vehicle {a['Ahead_Trend']}, {a['Gap_Trend']}{ttc}.")
    else:
        out.append("Ahead: no vehicle.")
    if isinstance(b, dict):
        out.append(f"Behind: {b['Gap_m']} m, {b['Behind_Trend']}.")
    else:
        out.append("Behind: no vehicle.")
    out.append(side("Left", s["Left_Lane"]) + ".")
    out.append(side("Right", s["Right_Lane"]) + ".")
    return " ".join(out)


def pick(scenarios, n):
    """take scenarios evenly from every event type so the test covers all situations"""
    random.seed(SEED)
    groups = {}
    for s in scenarios:
        groups.setdefault(s["Meta"]["Event_Type"], []).append(s)
    for g in groups.values():
        random.shuffle(g)
    chosen = []
    while len(chosen) < n and any(groups.values()):
        for g in groups.values():
            if g and len(chosen) < n:
                chosen.append(g.pop())
    return chosen


def main():
    with open(DATA_FILE, encoding="utf-8") as f:
        scenarios = [json.loads(line) for line in f if line.strip()]
    chosen = pick(scenarios, N_SCENARIOS)
    print(f"Scenarios in file: {len(scenarios)} | running: {len(chosen)} | model: {MODEL}" + (" | DRY RUN" if DRY else ""))

    results, total = [], 0.0
    for i, s in enumerate(chosen, 1):
        t0 = time.time()
        try:
            raw = ask_llama(s["Scenario"])
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            print("\nCould not reach Ollama at", OLLAMA_URL, "->", e)
            print("Make sure Ollama is running (open the Ollama app, or run 'ollama serve') and the model is downloaded ('ollama list').")
            sys.exit(1)
        secs = time.time() - t0
        total += secs
        try:
            answer, ok = json.loads(raw), True
        except json.JSONDecodeError:
            answer, ok = raw, False
        results.append({"Scenario_ID": s["Scenario_ID"], "event_type": s["Meta"]["Event_Type"], "seconds": round(secs, 1),
                        "valid_json": ok, "real_driver_next_3s": s["Meta"]["What_The_Real_Driver_Did_Next_3s"],
                        "input_to_llama": s["Scenario"], "answer": answer})
        print(f"[{i}/{len(chosen)}] {s['Scenario_ID']}  {s['Meta']['Event_Type']:<17} {secs:.1f}s  valid_json={ok}")

    with open("ngsim_pilot_results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open("ngsim_pilot_review.txt", "w", encoding="utf-8") as f:
        for r in results:
            f.write("=" * 70 + "\n")
            f.write(f"{r['Scenario_ID']}   type: {r['event_type']}   ({r['seconds']}s, valid_json={r['valid_json']})\n")
            f.write("Situation: " + summary(r["input_to_llama"]) + "\n")
            d = r["real_driver_next_3s"]
            f.write(f"What the real driver did in the next 3 s: speed change {d['Speed_Change_mps']} m/s, "
                    f"changed lane: {d['Changed_Lane']}\n\n")
            if r["valid_json"]:
                a = r["answer"]
                f.write("AV_Action: " + str(a.get("AV_Action")) + "\n")
                f.write("Urgency: " + str(a.get("Urgency")) + "\n")
                f.write("Reason: " + str(a.get("Reason")) + "\n")
                f.write("Safety_Consideration: " + str(a.get("Safety_Consideration")) + "\n")
            else:
                f.write("RAW (not valid JSON): " + str(r["answer"]) + "\n")
            f.write("\nYour review -> sensible? safe? uses only given facts?  ______\n\n")

    ok_count = sum(r["valid_json"] for r in results)
    print(f"\nDone. Valid JSON answers: {ok_count}/{len(results)} | average time: {total / max(len(results), 1):.1f}s")
    print("Saved: ngsim_pilot_results.jsonl and ngsim_pilot_review.txt")


if __name__ == "__main__":
    main()
