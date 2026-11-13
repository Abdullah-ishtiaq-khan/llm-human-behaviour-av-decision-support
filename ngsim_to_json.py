"""
ngsim_to_json.py - step 2 of the NGSIM project.

Reads us101_clean.csv (made by prepare_us101.py) and writes JSON scenarios an LLM can read:
    ngsim_scenarios.jsonl   one scenario per line
    ngsim_summary.txt       counts and simple statistics, for the report

Run:
    python ngsim_to_json.py            (300 scenarios)
    python ngsim_to_json.py 60         (small test with 60 scenarios)

How a scenario is made
  * We look at every vehicle once per second ("decision moment").
  * The vehicle is treated as the AV. We describe what it can sense at that moment:
    its own speed, the vehicle ahead, the vehicle behind and the lanes on both sides.
  * Units are converted from feet to metres.
  * We pick moments of different types so the data is not only easy, free-flowing traffic.
  * Each record has two parts:  "Scenario" (what the LLM sees)  and  "Meta" (for us only:
    event type, source ids, and what the real driver did next).
"""
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd

INPUT = "us101_clean.csv"
OUT_JSONL = "ngsim_scenarios.jsonl"
OUT_SUMMARY = "ngsim_summary.txt"
TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEED = 42
FT = 0.3048          # feet -> metres
FPS = 10             # NGSIM frames per second
MIN_GAP_M = 0.5      # reject scenarios where the data says vehicles overlap (noise)

# share of the scenarios for each event type
SHARES = {
    "close_following": 0.20,
    "leader_braking": 0.20,
    "lane_change": 0.20,
    "new_leader_close": 0.15,
    "on_ramp": 0.10,
    "normal": 0.15,
}
VTYPE = {1: "Motorcycle", 2: "Car", 3: "Truck"}


def lane_name(lane):
    lane = int(lane)
    if lane == 6:
        return "Auxiliary lane"
    if lane == 7:
        return "On-ramp lane"
    if lane == 8:
        return "Off-ramp lane"
    return f"Lane {lane} of 5 (1 = leftmost)"


# ------------------------------------------------------------------ load
print("Loading", INPUT, "...")
dt = {"Vehicle_ID": "int32", "Frame_ID": "int32", "Global_Time": "int64", "Local_X": "float32", "Local_Y": "float32",
      "v_length": "float32", "v_Width": "float32", "v_Class": "int8", "v_Vel": "float32", "v_Acc": "float32",
      "Lane_ID": "int8", "Preceding": "int32", "Following": "int32", "Space_Headway": "float32",
      "Time_Headway": "float32"}
df = pd.read_csv(INPUT, usecols=list(dt.keys()), dtype=dt)
df = df[df["Lane_ID"].between(1, 7)].copy()
# Frame_ID can restart in each recording period, so we use the clock time (0.1 s steps) as the time key
df["T"] = (df["Global_Time"] // 100).astype("int64")
df.sort_values(["Vehicle_ID", "T"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  {len(df):,} rows, {df['Vehicle_ID'].nunique():,} vehicles")


# which lanes exist where: a lane "exists" at a position if cars were seen in it within about 50 m of that position
_bins = (df["Lane_ID"].astype("int64") * 100000 + (df["Local_Y"] // 50).astype("int64")).unique()
LANE_BINS = set(_bins.tolist())


def lane_exists(lane, y_ft):
    b = int(y_ft // 50)
    return any((lane * 100000 + k) in LANE_BINS for k in (b - 1, b, b + 1))


# ------------------------------------------------------------------ history / future of each vehicle
g = df.groupby("Vehicle_ID", sort=False)


def shifted(col, k):
    """value of `col` k frames later (k<0: earlier), NaN if the vehicle has no such frame"""
    vals = g[col].shift(-k)
    ok = g["T"].shift(-k) == (df["T"] + k)
    return vals.where(ok)


df["v_p10"] = shifted("v_Vel", -10)       # speed 1 s ago
df["v_p30"] = shifted("v_Vel", -30)       # speed 3 s ago
df["v_n30"] = shifted("v_Vel", 30)        # speed 3 s later
df["lane_n30"] = shifted("Lane_ID", 30)   # lane 3 s later
df["prec_p20"] = shifted("Preceding", -20)  # leader 2 s ago

# ------------------------------------------------------------------ decision moments: 1 per second
sub = df[(df["T"] % FPS == 0) & df["v_p30"].notna() & df["v_n30"].notna()].copy()
print(f"  {len(sub):,} decision moments (1 per second per vehicle)")

lead = df[["T", "Vehicle_ID", "v_Vel", "v_p10", "v_Acc"]].rename(
    columns={"Vehicle_ID": "Preceding", "v_Vel": "lead_v", "v_p10": "lead_v_p10", "v_Acc": "lead_acc"})
sub = sub.merge(lead, on=["T", "Preceding"], how="left")

has_lead = (sub["Preceding"] > 0) & (sub["Space_Headway"] > 0) & sub["lead_v"].notna()
moving = sub["v_Vel"] > 15  # > 4.6 m/s

flags = pd.DataFrame(index=sub.index)
flags["close_following"] = has_lead & moving & (sub["Time_Headway"] > 0.1) & (sub["Time_Headway"] <= 1.0)
# NGSIM acceleration values are noisy (clipped at +-11.2 ft/s2), so braking is judged from the speed drop over 1 s (>= 2 m/s)
flags["leader_braking"] = has_lead & moving & ((sub["lead_v_p10"] - sub["lead_v"]) >= 6.5)
flags["lane_change"] = moving & sub["lane_n30"].notna() & (sub["lane_n30"] != sub["Lane_ID"]) & sub["Lane_ID"].between(1, 6) \
    & sub["lane_n30"].between(1, 6)
flags["new_leader_close"] = has_lead & moving & sub["prec_p20"].notna() & (sub["prec_p20"] > 0) \
    & (sub["prec_p20"] != sub["Preceding"]) & (sub["Space_Headway"] * FT < 20)
flags["on_ramp"] = moving & (sub["Lane_ID"] == 7)
flags["normal"] = moving & ~flags[["close_following", "leader_braking", "lane_change", "new_leader_close"]].any(axis=1) \
    & (sub["Lane_ID"] <= 5) & (((sub["Time_Headway"] >= 2.0) & (sub["Time_Headway"] <= 6.0) & has_lead) | (sub["Preceding"] == 0))
print("Candidate moments per event type:")
for k in SHARES:
    print(f"  {k:18s} {int(flags[k].sum()):>9,}")

# ------------------------------------------------------------------ frame snapshots
snap_df = df.sort_values("T")
frames = snap_df["T"].to_numpy()


def snapshot(f):
    a = np.searchsorted(frames, f, side="left")
    b = np.searchsorted(frames, f, side="right")
    return snap_df.iloc[a:b]


def rnd(x, n=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n) + 0.0


def lane_view(snap, ego, lane):
    """what is in the neighbouring lane next to the ego vehicle"""
    other = snap[snap["Lane_ID"] == lane]
    y_e, len_e = float(ego["Local_Y"]), float(ego["v_length"])
    best_ahead, best_behind, alongside = None, None, False
    for _, o in other.iterrows():
        y_o, len_o = float(o["Local_Y"]), float(o["v_length"])
        ahead = (y_o - len_o - y_e) * FT       # gap between ego front and the other's rear
        behind = (y_e - len_e - y_o) * FT      # gap between the other's front and ego rear
        if ahead < 0 and behind < 0:
            alongside = True
        elif ahead >= 0:
            if best_ahead is None or ahead < best_ahead[0]:
                best_ahead = (ahead, float(o["v_Vel"]) * FT)
        else:
            if best_behind is None or behind < best_behind[0]:
                best_behind = (behind, float(o["v_Vel"]) * FT)
    return {
        "Vehicle_Alongside": alongside,
        "Nearest_Vehicle_Ahead": None if best_ahead is None or best_ahead[0] > 100 else
        {"Gap_m": rnd(best_ahead[0]), "Speed_mps": rnd(best_ahead[1])},
        "Nearest_Vehicle_Behind": None if best_behind is None or best_behind[0] > 100 else
        {"Gap_m": rnd(best_behind[0]), "Speed_mps": rnd(best_behind[1])},
    }


def build(row, event):
    snap = snapshot(int(row["T"]))
    ego_rows = snap[snap["Vehicle_ID"] == row["Vehicle_ID"]]
    if ego_rows.empty:
        return None
    ego = ego_rows.iloc[0]
    v_e = float(ego["v_Vel"]) * FT
    y_e, len_e = float(ego["Local_Y"]), float(ego["v_length"])
    lane = int(ego["Lane_ID"])

    # vehicle ahead
    ahead = None
    if int(ego["Preceding"]) > 0:
        r = snap[snap["Vehicle_ID"] == ego["Preceding"]]
        if not r.empty:
            l = r.iloc[0]
            gap = (float(l["Local_Y"]) - float(l["v_length"]) - y_e) * FT
            if gap < MIN_GAP_M:
                return None
            v_l = float(l["v_Vel"]) * FT
            closing = v_e - v_l
            ahead = {
                "Vehicle_Type": VTYPE.get(int(l["v_Class"]), "Car"),
                "Gap_m": rnd(gap),
                "Speed_mps": rnd(v_l),
                "Speed_Change_Last_1s_mps": rnd((float(l["v_Vel"]) - float(row["lead_v_p10"])) * FT) if pd.notna(row["lead_v_p10"]) else None,
                "Relative_Speed_mps": rnd(closing),          # positive = AV is closing in
                "Time_Headway_s": rnd(gap / v_e) if v_e > 1 else None,
                "Time_To_Collision_s": rnd(gap / closing) if closing > 0.3 and gap / closing <= 30 else None,
            }
    # vehicle behind
    behind = None
    if int(ego["Following"]) > 0:
        r = snap[snap["Vehicle_ID"] == ego["Following"]]
        if not r.empty:
            f = r.iloc[0]
            gap = (y_e - len_e - float(f["Local_Y"])) * FT
            if gap < MIN_GAP_M:
                return None
            v_f = float(f["v_Vel"]) * FT
            closing = v_f - v_e
            behind = {
                "Vehicle_Type": VTYPE.get(int(f["v_Class"]), "Car"),
                "Gap_m": rnd(gap),
                "Speed_mps": rnd(v_f),
                "Relative_Speed_mps": rnd(closing),          # positive = it is closing in on the AV
                "Time_Headway_s": rnd(gap / v_f) if v_f > 1 else None,
                "Time_To_Collision_s": rnd(gap / closing) if closing > 0.3 and gap / closing <= 30 else None,
            }

    # vehicles within 100 m (any lane)
    near = snap[((snap["Local_Y"] - y_e).abs() * FT) <= 100]
    left = lane_view(snap, ego, lane - 1) if lane > 1 and lane_exists(lane - 1, y_e) else None
    right = lane_view(snap, ego, lane + 1) if lane < 7 and lane_exists(lane + 1, y_e) else None

    scenario = {
        "Road_Type": "Highway",
        "AV_Lane": lane_name(lane),
        "AV_Vehicle_Type": VTYPE.get(int(ego["v_Class"]), "Car"),
        "AV_Speed_mps": rnd(v_e),
        "AV_Speed_kmh": rnd(v_e * 3.6, 0),
        "AV_Acceleration_mps2": rnd(float(ego["v_Acc"]) * FT),
        "AV_Speed_Change_Last_3s_mps": rnd((float(ego["v_Vel"]) - float(row["v_p30"])) * FT),
        "Vehicle_Ahead": ahead,
        "Vehicle_Behind": behind,
        "Left_Lane": left if left is not None else "No lane on the left at this position",
        "Right_Lane": right if right is not None else "No lane on the right at this position",
        "Vehicles_Within_100m_All_Lanes": int(len(near) - 1),
    }
    meta = {
        "Source": "NGSIM US-101",
        "Event_Type": event,
        "Vehicle_ID": int(ego["Vehicle_ID"]),
        "Frame_ID": int(ego["Frame_ID"]),
        "Global_Time_ms": int(ego["Global_Time"]),
        "What_The_Real_Driver_Did_Next_3s": {
            "Speed_Change_mps": rnd((float(row["v_n30"]) - float(ego["v_Vel"])) * FT),
            "Changed_Lane": bool(row["lane_n30"] != ego["Lane_ID"]),
        },
    }
    return scenario, meta


# ------------------------------------------------------------------ pick scenarios
rng = np.random.RandomState(SEED)
used_vehicles = set()
records = []
quota = {k: max(1, int(round(TOTAL * s))) for k, s in SHARES.items()}
for event, n in quota.items():
    idx = sub.index[flags[event]].to_numpy().copy()
    rng.shuffle(idx)
    got = 0
    for i in idx:
        row = sub.loc[i]
        vid = int(row["Vehicle_ID"])
        if vid in used_vehicles:
            continue
        out = build(row, event)
        if out is None:
            continue
        used_vehicles.add(vid)
        records.append((out[0], out[1]))
        got += 1
        if got >= n:
            break
    print(f"  {event:18s} wanted {n:>3}  got {got:>3}")

rng.shuffle(records)
with open(OUT_JSONL, "w", encoding="utf-8") as fh:
    for k, (sc, meta) in enumerate(records, 1):
        fh.write(json.dumps({"Scenario_ID": f"N{k:04d}", "Meta": meta, "Scenario": sc}, ensure_ascii=False) + "\n")

# ------------------------------------------------------------------ summary
ev = Counter(m["Event_Type"] for _, m in records)
lines = [f"NGSIM US-101 scenarios: {len(records)}", "", "Scenarios per event type:"]
lines += [f"  {k}: {v}" for k, v in ev.items()]


def stats(name, vals):
    vals = [v for v in vals if v is not None]
    if vals:
        a = np.array(vals, dtype=float)
        lines.append(f"  {name}: n={len(a)}, min={a.min():.1f}, median={np.median(a):.1f}, max={a.max():.1f}")


lines += ["", "Value ranges (metres, seconds, m/s):"]
stats("AV speed (m/s)", [s["AV_Speed_mps"] for s, _ in records])
stats("Gap to vehicle ahead (m)", [s["Vehicle_Ahead"]["Gap_m"] for s, _ in records if s["Vehicle_Ahead"]])
stats("Time headway (s)", [s["Vehicle_Ahead"]["Time_Headway_s"] for s, _ in records if s["Vehicle_Ahead"]])
stats("Time-to-collision (s)", [s["Vehicle_Ahead"]["Time_To_Collision_s"] for s, _ in records if s["Vehicle_Ahead"]])
stats("Vehicles within 100 m", [s["Vehicles_Within_100m_All_Lanes"] for s, _ in records])
no_lead = sum(1 for s, _ in records if not s["Vehicle_Ahead"])
lines += ["", f"Scenarios with no vehicle ahead: {no_lead}",
          f"Scenarios where a vehicle was alongside on the left or right: "
          f"{sum(1 for s, _ in records if any(isinstance(s[k], dict) and s[k]['Vehicle_Alongside'] for k in ('Left_Lane', 'Right_Lane')))}"]
open(OUT_SUMMARY, "w", encoding="utf-8").write("\n".join(lines))
print("\n" + "\n".join(lines))
print(f"\nSaved {OUT_JSONL} and {OUT_SUMMARY}")
