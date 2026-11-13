"""
 small Llama pilot for the AV decision-support project.

"""
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

# ---------------- settings ----------------
DATA_FILE = "AV_dataset_clean.jsonl"
MODEL = "llama3.1:8b"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
N_SCENARIOS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEED = 42                      # same seed = same 20 scenarios every time
SKIP_SNOW_ICE = True           # snow/ice is not normal UAE driving
# Fields NOT given to Llama (labels/summaries an AV would not have). Edit this list if you change your mind.
DROP_FROM_PROMPT = ["Scenario_ID", "Scenario_Category", "Potential_Interaction_Type", "Scenario_Description"]
# ------------------------------------------

SYSTEM_PROMPT = """You are the decision-support module of an autonomous vehicle (AV).
You are given a structured description of a driving situation involving the AV and one human-driven vehicle.

Your task:
1. Analyse the road and environment conditions, the vehicle positions and speeds, the AV state, and the observable behaviour of the human-driven vehicle.
2. Recommend ONE appropriate HIGH-LEVEL action for the AV (for example: "Reduce speed and increase following distance"). Do NOT give low-level control commands such as brake percentages or steering angles.
3. Explain your reasoning briefly, using only the information provided.
4. State the main safety consideration.

Rules:
- Safety comes first.
- Do not invent facts that are not in the scenario.
- Answer ONLY with a JSON object with these keys: AV_Action, Urgency (Low, Medium or High), Reason, Safety_Consideration."""

SCHEMA = {
    "type": "object",
    "properties": {
        "AV_Action": {"type": "string"},
        "Urgency": {"type": "string", "enum": ["Low", "Medium", "High"]},
        "Reason": {"type": "string"},
        "Safety_Consideration": {"type": "string"},
    },
    "required": ["AV_Action", "Urgency", "Reason", "Safety_Consideration"],
}

ICE = {"icy", "ice patches", "slushy", "snow-covered"}


def is_snow_ice(s):
    return "snow" in str(s.get("Weather", "")).lower() or str(s.get("Road_Surface", "")).lower() in ICE


def ask_llama(scenario_for_prompt):
    user_prompt = "Scenario (JSON):\n" + json.dumps(scenario_for_prompt, indent=1, ensure_ascii=False) + \
                  "\n\nRespond with the JSON object only."
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user_prompt}],
        "stream": False,
        "format": SCHEMA,
        "options": {"temperature": 0.2, "num_ctx": 4096, "seed": SEED},
    }
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode("utf-8"))["message"]["content"]


def main():
    with open(DATA_FILE, encoding="utf-8") as f:
        scenarios = [json.loads(line) for line in f if line.strip()]
    pool = [s for s in scenarios if not (SKIP_SNOW_ICE and is_snow_ice(s))]
    random.seed(SEED)
    chosen = random.sample(pool, min(N_SCENARIOS, len(pool)))
    print(f"Scenarios in file: {len(scenarios)} | usable: {len(pool)} | running: {len(chosen)} | model: {MODEL}")

    results, total_time = [], 0.0
    for i, s in enumerate(chosen, 1):
        sid = s["Scenario_ID"]
        prompt_input = {k: v for k, v in s.items() if k not in DROP_FROM_PROMPT}
        t0 = time.time()
        try:
            raw = ask_llama(prompt_input)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            print("\nCould not reach Ollama at", OLLAMA_URL, "->", e)
            print("Make sure Ollama is running (open the Ollama app, or run 'ollama serve') and the model is downloaded ('ollama list').")
            sys.exit(1)
        secs = time.time() - t0
        total_time += secs
        try:
            answer, ok = json.loads(raw), True
        except json.JSONDecodeError:
            answer, ok = raw, False
        results.append({"Scenario_ID": sid, "seconds": round(secs, 1), "valid_json": ok,
                        "input_to_llama": prompt_input, "scenario_description": s.get("Scenario_Description"),
                        "answer": answer})
        print(f"[{i}/{len(chosen)}] {sid}  {secs:.1f}s  valid_json={ok}")

    with open("pilot_results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open("pilot_review.txt", "w", encoding="utf-8") as f:
        for r in results:
            f.write("=" * 70 + "\n")
            f.write(f"{r['Scenario_ID']}   ({r['seconds']}s, valid_json={r['valid_json']})\n")
            f.write("Scenario: " + str(r["scenario_description"]) + "\n\n")
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
    print(f"\nDone. Valid JSON answers: {ok_count}/{len(results)} | average time: {total_time / max(len(results), 1):.1f}s")
    print("Saved: pilot_results.jsonl and pilot_review.txt")


if __name__ == "__main__":
    main()
