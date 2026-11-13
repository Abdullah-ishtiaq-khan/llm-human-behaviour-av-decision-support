"""
server.py - the prototype's server (step 1 of 2: the Python side).

What it does
  - loads the 300 NGSIM scenarios (ngsim_scenarios.jsonl)
  - serves the web page (index.html, made in step 2)
  - for a scenario you pick, or values you type in the form, it runs the SAME pipeline as run_pilot_ngsim_v5.py:
        plain-word labels -> safety filter (allowed actions) -> Llama -> automatic check -> one retry if needed
    and sends the result back to the page.

Put this file in the same folder as run_pilot_ngsim_v5.py and ngsim_scenarios.jsonl, make sure Ollama is running, then:
    python server.py            (opens http://localhost:8000)
    python server.py 8080       (use another port)
    python server.py dry        (test without Llama: instant, dummy answers)
Stop it with Ctrl+C.
Uses only Python's standard library - nothing to install.
"""
import json
import os
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PORT, DRY = 8000, False
for _a in sys.argv[1:]:
    if _a == "dry":
        DRY = True
    elif _a.isdigit():
        PORT = int(_a)
sys.argv = [sys.argv[0]] + (["dry"] if DRY else [])     # the pilot script reads sys.argv when it is imported

try:
    import run_pilot_ngsim_v5 as pilot
except ImportError:
    sys.exit("Could not find run_pilot_ngsim_v5.py - put server.py in the same folder as run_pilot_ngsim_v5.py.")

DATA_FILE = os.path.join(HERE, "ngsim_scenarios.jsonl")
if not os.path.exists(DATA_FILE):
    sys.exit("Could not find ngsim_scenarios.jsonl - put it in the same folder as server.py.")
with open(DATA_FILE, encoding="utf-8") as _f:
    SCEN = {}
    for _line in _f:
        if _line.strip():
            _r = json.loads(_line)
            SCEN[_r["Scenario_ID"]] = _r
ORDER = sorted(SCEN)

LLAMA_LOCK = threading.Lock()          # one Llama request at a time (the laptop is CPU-only)
TREND_TO_CHANGE = {"slowing": -1.5, "steady": 0.0, "speeding up": 1.5}    # m/s over the last second
NO_LEFT, NO_RIGHT = "No lane on the left at this position", "No lane on the right at this position"


# ------------------------------------------------------------------ form <-> scenario
def rnd(x):
    return None if x is None else round(float(x), 1) + 0.0


def num(form, key, label, lo, hi):
    try:
        v = float(form.get(key))
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number")
    if not lo <= v <= hi:
        raise ValueError(f"{label} must be between {lo} and {hi}")
    return v


def side_dict(option, v_av):
    """option: none / free / alongside / close  ->  the same structure the converter writes"""
    if option == "free":
        return {"Vehicle_Alongside": False, "Nearest_Vehicle_Ahead": None, "Nearest_Vehicle_Behind": None}
    if option == "alongside":
        return {"Vehicle_Alongside": True, "Nearest_Vehicle_Ahead": {"Gap_m": 8.0, "Speed_mps": rnd(v_av)},
                "Nearest_Vehicle_Behind": {"Gap_m": 6.0, "Speed_mps": rnd(v_av)}}
    if option == "close":
        return {"Vehicle_Alongside": False, "Nearest_Vehicle_Ahead": {"Gap_m": 6.0, "Speed_mps": rnd(v_av)},
                "Nearest_Vehicle_Behind": {"Gap_m": 5.0, "Speed_mps": rnd(v_av)}}
    return None                                           # "none" = there is no such lane


def scenario_from_form(f):
    """build a scenario (same structure as the converter's) from the values typed in the form.
    Returns (scenario, notes)"""
    notes = []
    kmh = num(f, "av_speed_kmh", "AV speed", 0, 150)
    v_av = kmh / 3.6
    lane = str(f.get("lane", "3"))
    if lane not in ("1", "2", "3", "4", "5", "ramp"):
        raise ValueError("Lane must be 1-5 or the on-ramp")
    av_lane = "On-ramp lane" if lane == "ramp" else f"Lane {lane} of 5 (1 = leftmost)"

    ahead = None
    if f.get("ahead_exists"):
        gap = num(f, "ahead_gap_m", "Gap to the vehicle ahead", 0.5, 200)
        v_l = num(f, "ahead_speed_kmh", "Speed of the vehicle ahead", 0, 200) / 3.6
        trend = f.get("ahead_trend", "steady")
        if trend not in TREND_TO_CHANGE:
            raise ValueError("The vehicle ahead must be slowing, steady or speeding up")
        closing = v_av - v_l
        ahead = {"Vehicle_Type": "Car", "Gap_m": rnd(gap), "Speed_mps": rnd(v_l),
                 "Speed_Change_Last_1s_mps": TREND_TO_CHANGE[trend], "Relative_Speed_mps": rnd(closing),
                 "Time_Headway_s": rnd(gap / v_av) if v_av > 1 else None,
                 "Time_To_Collision_s": rnd(gap / closing) if closing > 0.3 and gap / closing <= 30 else None}
    behind = None
    if f.get("behind_exists"):
        gap = num(f, "behind_gap_m", "Gap to the vehicle behind", 0.5, 200)
        v_b = num(f, "behind_speed_kmh", "Speed of the vehicle behind", 0, 200) / 3.6
        closing = v_b - v_av
        behind = {"Vehicle_Type": "Car", "Gap_m": rnd(gap), "Speed_mps": rnd(v_b), "Relative_Speed_mps": rnd(closing),
                  "Time_Headway_s": rnd(gap / v_b) if v_b > 1 else None,
                  "Time_To_Collision_s": rnd(gap / closing) if closing > 0.3 and gap / closing <= 30 else None}

    left_opt, right_opt = f.get("left_lane", "none"), f.get("right_lane", "none")
    for o in (left_opt, right_opt):
        if o not in ("none", "free", "alongside", "close"):
            raise ValueError("Side lanes must be none, free, alongside or close")
    if lane == "1" and left_opt != "none":
        left_opt = "none"
        notes.append("Lane 1 is the leftmost lane, so there is no lane on its left.")
    if lane in ("5", "ramp") and right_opt != "none":
        right_opt = "none"
        notes.append("There is no lane on the right of lane 5 or of the on-ramp.")
    near = int(num(f, "traffic_100m", "Vehicles within 100 m", 0, 150))

    return {"Road_Type": "Highway", "AV_Lane": av_lane, "AV_Vehicle_Type": "Car",
            "AV_Speed_mps": rnd(v_av), "AV_Speed_kmh": rnd(kmh), "AV_Acceleration_mps2": 0.0,
            "AV_Speed_Change_Last_3s_mps": 0.0, "Vehicle_Ahead": ahead, "Vehicle_Behind": behind,
            "Left_Lane": side_dict(left_opt, v_av) or NO_LEFT, "Right_Lane": side_dict(right_opt, v_av) or NO_RIGHT,
            "Vehicles_Within_100m_All_Lanes": near}, notes


def form_from_scenario(raw):
    """the form values that describe a stored scenario (used to fill the form when a scenario is picked)"""
    lane = "ramp" if raw["AV_Lane"].lower().startswith("on-ramp") else raw["AV_Lane"].split()[1]
    a, b = raw["Vehicle_Ahead"], raw["Vehicle_Behind"]

    def side(x):
        if not isinstance(x, dict):
            return "none"
        if x["Vehicle_Alongside"]:
            return "alongside"
        return "free" if pilot.side_label(x)["Space_For_Lane_Change"] == "yes" else "close"

    trend = {"slowing down": "slowing", "speeding up": "speeding up", "steady": "steady"}
    return {"av_speed_kmh": raw["AV_Speed_kmh"], "lane": lane,
            "ahead_exists": bool(a), "ahead_gap_m": a["Gap_m"] if a else 20,
            "ahead_speed_kmh": round(a["Speed_mps"] * 3.6) if a else round(raw["AV_Speed_kmh"]),
            "ahead_trend": trend.get(pilot.trend(a["Speed_Change_Last_1s_mps"]), "steady") if a else "steady",
            "behind_exists": bool(b), "behind_gap_m": b["Gap_m"] if b else 20,
            "behind_speed_kmh": round(b["Speed_mps"] * 3.6) if b else round(raw["AV_Speed_kmh"]),
            "left_lane": side(raw["Left_Lane"]), "right_lane": side(raw["Right_Lane"]),
            "traffic_100m": raw["Vehicles_Within_100m_All_Lanes"]}


# ------------------------------------------------------------------ running the pipeline
def split_flag(flag):
    what, _, sentence = flag.partition("  <- ")
    return {"rule": what[:1], "problem": what[3:] if what[1:2] == ":" else what, "sentence": sentence.strip('"')}


def recommend(raw, sid, event, real):
    item = {"Scenario_ID": sid, "Meta": {"Event_Type": event, "What_The_Real_Driver_Did_Next_3s":
            real or {"Speed_Change_mps": None, "Changed_Lane": None}}, "Scenario": raw}
    with LLAMA_LOCK:
        rec = pilot.answer_one(item)
    return {"scenario_id": sid, "event_type": event, "summary": pilot.summary(raw), "labels": pilot.label(raw),
            "allowed_actions": rec["allowed_actions"], "allowed_urgency": rec["allowed_urgency"],
            "answer": rec["answer"], "valid_json": rec["valid_json"], "attempts": rec["attempts"],
            "first_answer": rec.get("first_answer"), "first_problems": [split_flag(x) for x in rec.get("first_flags", [])],
            "problems": [split_flag(x) for x in rec["flags"]], "seconds": rec["seconds"], "real_driver": real}


def ollama_up():
    if DRY:
        return None
    try:
        urllib.request.urlopen(pilot.OLLAMA_URL + "/api/tags", timeout=2)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ web server
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):                  # keep the terminal quiet
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_page(self, html, code=200):
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            p = os.path.join(HERE, "index.html")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    return self.send_page(f.read())
            return self.send_page("<h2>The server is running.</h2><p>index.html was not found yet "
                                  "(it comes in step 2). You can already test the data: "
                                  "<a href='/api/scenarios'>/api/scenarios</a>, "
                                  "<a href='/api/scenario/N0153'>/api/scenario/N0153</a>, "
                                  "<a href='/api/health'>/api/health</a>.</p>")
        if path == "/api/health":
            return self.send_json({"ok": True, "dry_run": DRY, "model": pilot.MODEL, "ollama_running": ollama_up(),
                                   "scenarios": len(SCEN)})
        if path == "/api/scenarios":
            return self.send_json([{"id": sid, "type": SCEN[sid]["Meta"]["Event_Type"],
                                    "title": f"{sid} - {SCEN[sid]['Meta']['Event_Type'].replace('_', ' ')} - "
                                             f"{SCEN[sid]['Scenario']['AV_Speed_kmh']:.0f} km/h"} for sid in ORDER])
        if path.startswith("/api/scenario/"):
            sid = path.rsplit("/", 1)[1]
            if sid not in SCEN:
                return self.send_json({"error": f"No scenario called {sid}"}, 404)
            r = SCEN[sid]
            return self.send_json({"id": sid, "type": r["Meta"]["Event_Type"], "summary": pilot.summary(r["Scenario"]),
                                   "form": form_from_scenario(r["Scenario"]),
                                   "real_driver": r["Meta"]["What_The_Real_Driver_Did_Next_3s"]})
        self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/recommend":
            return self.send_json({"error": "Not found"}, 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self.send_json({"error": "The request was not valid JSON"}, 400)
        try:
            notes = []
            if isinstance(body.get("form"), dict):                       # values typed or edited in the form
                raw, notes = scenario_from_form(body["form"])
                sid, event, real = body.get("scenario_id") or "custom", "custom", None
                if sid in SCEN:
                    sid = sid + " (edited)"
            elif body.get("scenario_id") in SCEN:                         # a stored scenario, unchanged
                r = SCEN[body["scenario_id"]]
                raw, sid, event = r["Scenario"], body["scenario_id"], r["Meta"]["Event_Type"]
                real = r["Meta"]["What_The_Real_Driver_Did_Next_3s"]
            else:
                return self.send_json({"error": "Send a scenario_id or the form values"}, 400)
        except ValueError as e:
            return self.send_json({"error": str(e)}, 400)
        try:
            out = recommend(raw, sid, event, real)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            return self.send_json({"error": f"Could not reach Ollama at {pilot.OLLAMA_URL}. Open the Ollama app "
                                            f"(or run 'ollama serve') and try again. ({e})"}, 503)
        out["notes"] = notes
        self.send_json(out)


def main():
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        sys.exit(f"Port {PORT} is already in use. Close the other window, or run: python server.py {PORT + 1}")
    url = f"http://localhost:{PORT}"
    print(f"Prototype server running at {url}   (model: {pilot.MODEL}{', DRY RUN - no Llama' if DRY else ''})")
    print(f"{len(SCEN)} scenarios loaded. Press Ctrl+C to stop.")
    if not DRY and not ollama_up():
        print("WARNING: Ollama does not seem to be running. Open the Ollama app before you ask for a recommendation.")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
