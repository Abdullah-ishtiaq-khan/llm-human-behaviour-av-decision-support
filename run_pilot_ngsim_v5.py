"""
run_pilot_ngsim_v5.py - Llama on the NGSIM scenarios: rule-based safety filter + automatic check and one retry.

For every scenario:
  1. a small set of rules decides which actions are ALLOWED (safety filter)
  2. Llama picks one allowed action and explains it
  3. the answer is checked against the scenario (same checks as check_answers.py)
  4. if the explanation contradicts the data, Llama is told what was wrong and answers once more

Put this file in the same folder as ngsim_scenarios.jsonl, make sure Ollama is running, then:
    python run_pilot_ngsim_v5.py            (20 scenarios)
    python run_pilot_ngsim_v5.py 5          (quick test)
    python run_pilot_ngsim_v5.py 300        (all scenarios, about 3 hours - safe to stop and restart)

Files written (the number is the run size, so a 20-run and a 300-run never overwrite each other):
    ngsim_results_N.jsonl   full results, saved after EVERY scenario
    ngsim_review_N.txt      easy-to-read version for checking by hand
If the run is stopped (laptop sleeps, Ctrl+C) just run the same command again: it continues where it stopped.
Add the word  fresh  to throw away the old results of that run size and start over, e.g.  python run_pilot_ngsim_v5.py 20 fresh
Only the "Scenario" part is sent to Llama. The "Meta" part (event type, what the real driver did next) is kept for you.
Uses only Python's standard library - nothing to install.
"""
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

DATA_FILE = "ngsim_scenarios.jsonl"
MODEL = "llama3.1:8b"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
args = [a for a in sys.argv[1:] if a not in ("dry", "fresh")]
DRY = "dry" in sys.argv[1:]            # "python run_pilot_ngsim_v5.py 5 dry" = test the script without Llama
FRESH = "fresh" in sys.argv[1:]
N_SCENARIOS = int(args[0]) if args else 20
SEED = 42
RESULTS_FILE = f"ngsim_results_{N_SCENARIOS}.jsonl"
REVIEW_FILE = f"ngsim_review_{N_SCENARIOS}.txt"

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
- Ahead_Trend, Behind_Trend and Gap_Trend are already worked out for you. Use these words; do not recalculate them.
- Following_Gap: "very short" (under 1 s) is tailgating, "short" is 1 to 2 s, "comfortable" is over 2 s.
- Collision_Risk comes from the time-to-collision: "critical" (under 4 s), "elevated" (4 to 8 s), "none" otherwise.
- Space_For_Lane_Change is "yes" only when the lane exists, nobody is alongside and the gaps are large enough.
- Merge_Lane = true means the AV is on the on-ramp and has to join the main lanes.

A safety filter has already removed the actions that are not safe for this situation. The scenario contains Allowed_Actions and Allowed_Urgency. Choose the ONE allowed action that best fits the situation and one allowed urgency level.

Rules:
- Use only facts that are in the scenario. Do not mention a lane position, a risk or a vehicle that is not given.
- Write the reason in your own words, one or two sentences.
- Do not give low-level commands such as brake percentages or steering angles.

Answer ONLY with a JSON object with these keys: AV_Action, Urgency, Reason, Safety_Consideration (one sentence)."""


def make_schema(allowed, urgencies):
    return {
        "type": "object",
        "properties": {
            "AV_Action": {"type": "string", "enum": allowed},
            "Urgency": {"type": "string", "enum": urgencies},
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


def safety_filter(L):
    """rules that decide which actions are allowed. L = the labelled scenario (output of label()).
    Returns (allowed_actions, allowed_urgency)."""
    a, b = L["Vehicle_Ahead"], L["Vehicle_Behind"]
    left_ok = L["Left_Lane"].get("Space_For_Lane_Change") == "yes"
    right_ok = L["Right_Lane"].get("Space_For_Lane_Change") == "yes"
    lanes = []
    if left_ok:
        lanes.append("Prepare to change lane to the left")
    if right_ok:
        lanes.append("Prepare to change lane to the right")

    if not isinstance(a, dict):                                   # nothing ahead
        if L["Merge_Lane"]:
            return ["Accelerate and merge into the main lanes", "Accelerate gently"], ["Low", "Medium"]
        return ["Maintain speed and lane", "Accelerate gently"], ["Low"]

    risk, gap, tr = a["Collision_Risk"], a["Following_Gap"], a["Ahead_Trend"]
    if risk == "critical" or (gap == "very short" and tr == "slowing down"):
        return ["Brake firmly", "Reduce speed and increase following distance"], ["High"]
    if gap == "very short" or risk == "elevated":
        return ["Reduce speed and increase following distance", "Reduce speed slightly"] + lanes, ["Medium", "High"]
    if gap == "short" and tr == "slowing down":
        return ["Reduce speed slightly", "Reduce speed and increase following distance"] + lanes, ["Low", "Medium"]
    if gap is None:                                               # AV almost stopped: no headway value, stay careful
        return ["Maintain speed and lane", "Reduce speed slightly"], ["Low", "Medium"]
    # comfortable gap, or a short gap with a steady / speeding-up leader and no risk
    opts = ["Maintain speed and lane"]
    if isinstance(b, dict) and b["Behind_Trend"] == "vehicle behind is closing in" and b["Time_Headway_s"] is not None and b["Time_Headway_s"] < 1.0:
        opts += lanes                                              # being closed in on from behind: moving over is an option
    if gap == "short" or tr == "slowing down":
        opts.append("Reduce speed slightly")
    else:
        opts.append("Accelerate gently")
    return opts, ["Low", "Medium"]


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
      "Reason": "The car in front is losing speed while only 0.7 s ahead, and the collision risk is critical.",
      "Safety_Consideration": "Slowing early gives the vehicle behind time to react as well."}),
    (_ex("On-ramp lane", 45.0, None, None, NO_L, _lane(False, 60.0, 50.0), 20),
     {"AV_Action": "Accelerate and merge into the main lanes", "Urgency": "Medium",
      "Reason": "Nothing is ahead or behind on the ramp and the lane beside it is free, so the AV can speed up to traffic speed and join.",
      "Safety_Consideration": "Confirm the gap in the main lane before moving across."}),
    (_ex("Lane 2 of 5 (1 = leftmost)", 50.0, _veh(35.0, 14.0, 0.2, -0.1, 2.5),
         _veh(30.0, 14.0, 0, 0.0, 2.1), _lane(False, 50.0, 40.0), _lane(True, 20.0, 15.0), 25),
     {"AV_Action": "Maintain speed and lane", "Urgency": "Low",
      "Reason": "The gap ahead is comfortable, the leader is steady and nothing behind is approaching, so there is no need to change anything.",
      "Safety_Consideration": "Keep the same distance to the car ahead."}),
    (_ex("Lane 4 of 5 (1 = leftmost)", 58.0, _veh(8.0, 16.0, 0.0, 0.1, 0.5),
         _veh(25.0, 16.0, 0, 0.0, 1.6), _lane(False, 45.0, 30.0), _lane(True, 12.0, 9.0), 28),
     {"AV_Action": "Prepare to change lane to the left", "Urgency": "Medium",
      "Reason": "The AV is following very closely behind a steady vehicle and the left lane has a large free space.",
      "Safety_Consideration": "Check the left lane again just before moving over."}),
]


def prepared(scenario):
    """labelled scenario + the allowed actions from the safety filter"""
    L = label(scenario)
    allowed, urg = safety_filter(L)
    L = dict(L)
    L["Allowed_Actions"] = allowed
    L["Allowed_Urgency"] = urg
    return L, allowed, urg


def build_messages(scenario):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_in, ex_out in EXAMPLES:
        L, allowed, urg = prepared(ex_in)
        msgs.append({"role": "user", "content": "Scenario (JSON):\n" + json.dumps(L, ensure_ascii=False)})
        msgs.append({"role": "assistant", "content": json.dumps(ex_out, ensure_ascii=False)})
    L, allowed, urg = prepared(scenario)
    msgs.append({"role": "user", "content": "Scenario (JSON):\n" + json.dumps(L, ensure_ascii=False)
                 + "\n\nRespond with the JSON object only."})
    return msgs, allowed, urg


def ask_llama(scenario, history=None):
    """history = extra chat messages (used for the retry: Llama's first answer + what was wrong with it)"""
    msgs, allowed, urg = build_messages(scenario)
    if DRY:
        return json.dumps({"AV_Action": allowed[0], "Urgency": urg[0], "Reason": "(dry run)", "Safety_Consideration": "-"}), allowed, urg
    if history:
        msgs = msgs + history
    body = {
        "model": MODEL,
        "messages": msgs,
        "stream": False,
        "format": make_schema(allowed, urg),
        "options": {"temperature": 0.2, "num_ctx": 6144, "seed": SEED},
    }
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode("utf-8"))["message"]["content"], allowed, urg


# ---------------------------------------------------------------- automatic check of an answer
NEG = re.compile(r"\b(not|no|isn't|is not|aren't|n't|nor|neither|without)\b", re.I)


def _sentences(text):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", text or "") if x.strip()]


# phrases that tie a statement to the vehicle AHEAD / BEHIND (not to the AV itself)
_AHEAD = r"(?:(?:vehicle|car|truck|motorcycle|traffic|one)\s+(?:ahead|in front)|leader|lead vehicle|front vehicle)"
_BEHIND = r"(?:(?:vehicle|car|truck|motorcycle|traffic|one)\s+behind|following vehicle|vehicle in the rear)"


def _mentions_ahead(t):
    return re.search(_AHEAD, t, re.I) is not None


def _mentions_behind(t):
    return re.search(_BEHIND, t, re.I) is not None


def check_one(rec):
    """rec needs: valid_json, answer, input_to_llama (raw scenario), allowed_actions, allowed_urgency.
    Returns a list of flags (empty list = no contradiction found)."""
    flags = []
    ans = rec["answer"]
    if not rec["valid_json"] or not isinstance(ans, dict):
        return ["answer is not valid JSON"]
    L = label(rec["input_to_llama"])
    a, b = L["Vehicle_Ahead"], L["Vehicle_Behind"]
    action, urg = ans.get("AV_Action", ""), ans.get("Urgency", "")

    # A - allowed by the filter
    if action not in rec["allowed_actions"]:
        flags.append(f"A: action '{action}' is not in the allowed list")
    if urg not in rec["allowed_urgency"]:
        flags.append(f"A: urgency '{urg}' is not in the allowed list")
    # B - critical risk
    if isinstance(a, dict) and a["Collision_Risk"] == "critical":
        if action not in ("Brake firmly", "Reduce speed and increase following distance") or urg != "High":
            flags.append("B: critical collision risk but the answer is not a firm slow-down with High urgency")
    # C - lane change needs space
    if "change lane to the left" in action and L["Left_Lane"].get("Space_For_Lane_Change") != "yes":
        flags.append("C: lane change to the left without space")
    if "change lane to the right" in action and L["Right_Lane"].get("Space_For_Lane_Change") != "yes":
        flags.append("C: lane change to the right without space")
    # D - merge only on the ramp
    if "merge" in action.lower() and not L["Merge_Lane"]:
        flags.append("D: merge action while not on the on-ramp")

    # E - text consistency (a statement about the vehicle ahead / behind must match the data)
    for sent in _sentences(ans.get("Reason", "")) + _sentences(ans.get("Safety_Consideration", "")):
        low = sent.lower()
        neg = NEG.search(sent) is not None
        behind_closing = isinstance(b, dict) and "closing" in (b["Behind_Trend"] or "")
        if re.search(_BEHIND + r"[^.]*?(closing in|approaching|catching up|tailgating|too close)", low) and not neg and not behind_closing:
            if isinstance(b, dict):
                flags.append(f"E: says the vehicle behind is closing in, data: '{b['Behind_Trend']}'  <- \"{sent}\"")
            else:
                flags.append(f"E: talks about a vehicle behind but there is none  <- \"{sent}\"")
        if _mentions_ahead(sent) and not isinstance(a, dict) and not neg:
            flags.append(f"E: talks about a vehicle ahead but there is none  <- \"{sent}\"")
        if isinstance(a, dict):
            if re.search(_AHEAD + r"\s+(?:is |are |was |has been )?(?:slowing|braking|decelerating|losing speed)", low) \
                    and not neg and a["Ahead_Trend"] != "slowing down":
                flags.append(f"E: says the vehicle ahead is slowing, data: '{a['Ahead_Trend']}'  <- \"{sent}\"")
            if re.search(_AHEAD + r"\s+(?:is |are |was |has been )?(?:speeding up|accelerating|gaining speed)", low) \
                    and not neg and a["Ahead_Trend"] != "speeding up":
                flags.append(f"E: says the vehicle ahead is speeding up, data: '{a['Ahead_Trend']}'  <- \"{sent}\"")
            if re.search(r"\bAV\b.*\b(approaching|closing in on|too quickly)\b", sent) and not neg \
                    and a["Gap_Trend"] != "AV is closing in":
                flags.append(f"E: says the AV is approaching the vehicle ahead, data: '{a['Gap_Trend']}'  <- \"{sent}\"")
        m = re.search(r"collision risk (?:is|of|being)\s+(critical|elevated|none|high|low|moderate)", low)
        if m:
            real = a["Collision_Risk"] if isinstance(a, dict) else "none"
            said = m.group(1)
            if said in ("high", "critical") and real != "critical" or said == "elevated" and real != "elevated" \
                    or said in ("none", "low") and real != "none":
                flags.append(f"E: says collision risk is '{said}', data: '{real}'  <- \"{sent}\"")
        for side_name, key in (("left", "Left_Lane"), ("right", "Right_Lane")):
            if re.search(rf"\b{side_name} lane\b", low) and re.search(r"free|available|large|clear|open|space", low) and not neg \
                    and L[key].get("Space_For_Lane_Change") != "yes":
                flags.append(f"E: calls the {side_name} lane free, but it has no space or does not exist  <- \"{sent}\"")
        if re.search(r"middle lane|centre lane|center lane", low):
            flags.append(f"E: mentions a 'middle lane' that is not in the data  <- \"{sent}\"")
    return flags


def retry_history(raw, flags):
    """messages that tell Llama what was wrong with its first answer"""
    problems = "\n".join("- " + f.split("  <-")[0][3:] for f in flags)
    return [{"role": "assistant", "content": raw},
            {"role": "user", "content": "Your answer has mistakes compared with the scenario:\n" + problems +
             "\nAnswer again in the same JSON format. Use only the facts in the scenario, and choose only from the allowed actions."}]


def answer_one(s):
    """full processing of one scenario: ask, check, retry once if the explanation contradicts the data"""
    t0 = time.time()
    raw, allowed, urg = ask_llama(s["Scenario"])
    rec = {"Scenario_ID": s["Scenario_ID"], "event_type": s["Meta"]["Event_Type"],
           "real_driver_next_3s": s["Meta"]["What_The_Real_Driver_Did_Next_3s"],
           "allowed_actions": allowed, "allowed_urgency": urg, "input_to_llama": s["Scenario"], "attempts": 1}
    try:
        rec["answer"], rec["valid_json"] = json.loads(raw), True
    except json.JSONDecodeError:
        rec["answer"], rec["valid_json"] = raw, False
    rec["flags"] = check_one(rec)
    if rec["valid_json"] and rec["flags"] and not DRY:
        rec["first_answer"], rec["first_flags"] = rec["answer"], rec["flags"]
        raw2, _, _ = ask_llama(s["Scenario"], retry_history(raw, rec["flags"]))
        rec["attempts"] = 2
        try:
            rec["answer"], rec["valid_json"] = json.loads(raw2), True
        except json.JSONDecodeError:
            rec["answer"], rec["valid_json"] = rec["first_answer"], True      # keep the first answer if the retry is broken
        rec["flags"] = check_one(rec)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


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


def write_review(results):
    with open(REVIEW_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write("=" * 70 + "\n")
            f.write(f"{r['Scenario_ID']}   type: {r['event_type']}   ({r['seconds']}s, valid_json={r['valid_json']}, attempts={r['attempts']})\n")
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
            f.write("(allowed by the safety filter: " + "; ".join(r["allowed_actions"]) + " | urgency: " + "/".join(r["allowed_urgency"]) + ")\n")
            if r.get("first_flags"):
                f.write("(first answer was wrong, retried: " + " | ".join(x.split("  <-")[0] for x in r["first_flags"]) + ")\n")
            if r["flags"]:
                f.write("(STILL FLAGGED by the checker: " + " | ".join(x.split("  <-")[0] for x in r["flags"]) + ")\n")
            f.write("\nYour review -> sensible? safe? uses only given facts?  ______\n\n")


def main():
    with open(DATA_FILE, encoding="utf-8") as f:
        scenarios = [json.loads(line) for line in f if line.strip()]
    chosen = pick(scenarios, N_SCENARIOS)

    done = {}
    if os.path.exists(RESULTS_FILE) and not FRESH:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["Scenario_ID"]] = r
    print(f"Scenarios in file: {len(scenarios)} | running: {len(chosen)} | model: {MODEL}" + (" | DRY RUN" if DRY else ""))
    if done:
        print(f"Continuing an earlier run: {len(done)} already done and kept. (add the word 'fresh' to start over)")

    new_secs, new_n = 0.0, 0
    with open(RESULTS_FILE, "a" if done else "w", encoding="utf-8") as fout:
        for i, s in enumerate(chosen, 1):
            if s["Scenario_ID"] in done:
                continue
            try:
                rec = answer_one(s)
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                print("\nCould not reach Ollama at", OLLAMA_URL, "->", e)
                print("Make sure Ollama is running (open the Ollama app, or run 'ollama serve') and the model is downloaded ('ollama list').")
                print("Everything finished so far is saved. Run the same command again to continue.")
                sys.exit(1)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done[rec["Scenario_ID"]] = rec
            new_secs += rec["seconds"]
            new_n += 1
            note = "  (retried)" if rec["attempts"] == 2 else ""
            note += "  STILL FLAGGED" if rec["flags"] else ""
            print(f"[{i}/{len(chosen)}] {s['Scenario_ID']}  {s['Meta']['Event_Type']:<17} {rec['seconds']:.1f}s  valid_json={rec['valid_json']}{note}")

    results = [done[s["Scenario_ID"]] for s in chosen if s["Scenario_ID"] in done]
    write_review(results)
    ok_count = sum(r["valid_json"] for r in results)
    retried = sum(r["attempts"] == 2 for r in results)
    flagged = sum(bool(r["flags"]) for r in results)
    print(f"\nDone. Valid JSON answers: {ok_count}/{len(results)} | retried: {retried} | still flagged by the checker: {flagged}"
          f" | average time this session: {new_secs / max(new_n, 1):.1f}s")
    print(f"Saved: {RESULTS_FILE} and {REVIEW_FILE}")
    print(f"Next: python check_answers.py {RESULTS_FILE}")


if __name__ == "__main__":
    main()
