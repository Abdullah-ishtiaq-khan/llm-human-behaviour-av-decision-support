# LLM-Based Human Behaviour-Aware Decision-Making for Autonomous Vehicles

**Junior Project | Computer Science | University of Sharjah**

**Team Members:**
- Abdullah Ishtiaq Khan
- Hydyr Babamuradov
- Muhammed Adnan

**Supervisors:** Dr. Ibrahim, Dr. Bouziane

---

## Overview

This project develops a decision-support system for an autonomous vehicle (AV) travelling on a motorway.

The system takes real traffic situations from the **NGSIM US-101 dataset**, converts numerical measurements into plain-language descriptions, applies a rule-based safety filter, and asks a locally hosted **Llama 3.1 8B** model (through Ollama) to select a driving action and explain its decision.

An automatic checker then examines the model's explanation against the available traffic data. If a contradiction is detected, the model receives feedback and is allowed one retry.

**Important:** The system recommends and explains driving actions. It does not control a vehicle. The Llama model is used without training or fine-tuning.

### System Pipeline

```text
NGSIM US-101 Dataset
        |
        v
  300 Traffic Scenarios
        |
        v
   Plain-Word Labels
        |
        v
 Safety Filter (F1-F7)
        |
        v
   Llama 3.1 8B
   (Local via Ollama)
        |
        v
 Automatic Checker (A-E)
        |
        v
  Retry Once if Flagged
        |
        v
 Recommendation + Explanation
        |
        v
     Web Prototype
```

---

## Main Features

- **300 traffic scenarios:** Built from real NGSIM traffic data, covering six types of traffic situations.
- **Plain-language labels:** Converts numerical measurements into understandable descriptions, including gap, vehicle movement trends, collision risk and lane-change space.
- **Rule-based safety filter:** Uses rules F1-F7 to determine which of the eight driving actions and urgency levels are allowed.
- **Local AI model:** Uses Llama 3.1 8B through Ollama to generate a recommendation, a short explanation and a safety consideration.
- **Automatic checking:** Checks every answer using rules A-E and allows one retry if a problem is detected.
- **Web prototype:** Allows users to select one of the 300 scenarios or enter a custom traffic situation.
- **Evaluation:** Assesses the system through human review, comparison with real-driver behaviour and an independent Gemini review.

All model inference is performed locally. No cloud LLM API is required.

---

## Project Structure

| File / Folder | Description |
|---|---|
| `prepare_us101.py` | Cleans the raw NGSIM file, keeps US-101 data, reads it in chunks and converts feet to metres. |
| `ngsim_to_json.py` | Builds the 300 scenarios using fixed quotas, seed 42 and one scenario per vehicle. |
| `ngsim_scenarios.jsonl` | Contains the 300 scenarios. Scenario information is sent to Llama, while metadata about the event type and real driver behaviour is kept separate. |
| `run_pilot_ngsim_v5.py` | Main pipeline: labels, safety filter, Llama, checker and retry. |
| `check_answers.py` | Stand-alone checker using rules A-E. |
| `server.py` | Local web server and backend API using Python's standard library (port 8000). |
| `index.html` | Web interface for the prototype. |
| `make_evaluation_charts.py` | Generates evaluation charts E1-E7 and `evaluation_summary.txt`. |
| `results/` | Contains the final results, checker reports, reviews, evaluation summaries and figures. |
| `archive/` | Earlier versions and previous reports. |
| `data/` | Instructions for downloading the raw dataset. |

---

## Running the Prototype

### 1. Install Ollama

Install [Ollama](https://ollama.com/) and download the model:

```bash
ollama pull llama3.1:8b
```

Keep Ollama running.

### 2. Start the Web Server

Open a terminal in the project folder and run:

```bash
python server.py
```

### 3. Open the Prototype

Open the following address in your browser:

```text
http://localhost:8000
```

The server and main pipeline use the Python standard library.

`requirements.txt` (Pandas and NumPy) is needed only when rebuilding scenarios from the raw dataset.

The evaluation chart script also requires Matplotlib:

```bash
pip install matplotlib numpy
```

### Performance

On an ordinary CPU-only computer, one answer takes approximately 30 seconds, with longer times when a retry is required. The first request after starting Ollama is slower.

The server handles one Llama request at a time.

---

## Re-running the Experiment

The following commands can be used to run and evaluate the experiment.

### Run a Quick Test

```bash
python run_pilot_ngsim_v5.py 5
```

Runs five scenarios.

### Run All 300 Scenarios

```bash
python run_pilot_ngsim_v5.py 300
```

Runs all 300 scenarios. This takes approximately three hours on the tested computer.

### Run Without Calling Llama

```bash
python run_pilot_ngsim_v5.py 5 dry
```

Tests the script without making calls to Llama.

### Check Results

```bash
python check_answers.py ngsim_results_300.jsonl
```

### Rebuild Evaluation Charts

```bash
python make_evaluation_charts.py
```

Results are saved after every scenario, allowing a stopped experiment to continue without losing completed results.

---

## Safety Filter and Automatic Checking

### Safety Filter (Before Llama)

The safety filter checks the labelled traffic situation and determines which of the eight possible actions and urgency levels are permitted.

Examples:

- **Critical collision risk:** Only `Brake firmly` or `Reduce speed and increase following distance` are allowed, with High urgency.
- **On-ramp merging:** `Accelerate and merge` is allowed when there is no vehicle ahead and the AV is on the on-ramp.
- **Lane changes:** A lane change is offered only when the target lane has sufficient space.

The filter does not select the final action. It only removes actions that are not permitted. Llama chooses from the remaining allowed actions.

### Automatic Checker (After Llama)

The automatic checker examines the model's response using five rules:

| Rule | What it checks |
|---|---|
| A | The selected action and urgency are on the allowed list. |
| B | Critical collision risk receives a firm slow-down with High urgency. |
| C | A lane change is selected only when the target side has space. |
| D | A merge is selected only when the AV is on the on-ramp. |
| E | The sentences in `Reason` and `Safety_Consideration` do not contradict the traffic data. |

Rule E checks statements about vehicles ahead and behind, AV movement, collision-risk descriptions, free lanes and middle-lane information. Sentences containing negation are skipped.

### Retry Mechanism

If a rule identifies a problem in the first answer, Llama receives the specific complaint and is asked to generate another answer.

Only one retry is allowed. If the retry fails, the first answer is kept.

---

## Results (300 Scenarios)

| Measure | Result |
|---|---:|
| Valid JSON answers | 300 / 300 |
| Passed automatic checking on first attempt | 287 / 300 (95.7%) |
| Passed automatic checking after one retry | 298 / 300 (99.3%) |
| Scenarios requiring a retry | 43 (14.3%) |
| Average time per answer | 33.0 seconds |
| Median time per answer | 30.5 seconds |
| Maximum time per answer | 118.3 seconds |
| Critical-risk cases (TTC under 4 seconds) where Llama advised slowing down | 24 / 24 |

### Three Evaluations of Recommendation Quality

| Evaluation | Sensible | Safe | Uses only given facts |
|---|---:|---:|---:|
| 1. Human review (20 answers) | 20 / 20 (100%) | 19 / 20 (95%) | 16 / 20 (80%) |
| 2. Gemini review (same 20 answers) | 17 / 20 (85%) | 14 / 20 (70%) | 15 / 20 (75%) |
| 3. Gemini review (all 300 answers) | 276 (92%) | 256 (85%) | 231 (77%) |

### Comparison with Real Drivers

The second evaluation also compares Llama's recommendations with what real drivers did during the following three seconds.

This is a sanity check rather than an accuracy score because real drivers do not always make safe or correct decisions.

### Important Interpretation of the Results

The 99.3% result means that the automatic checker found no contradiction of the types it knows in 298 of 300 final answers. It does **not** mean that 99.3% of explanations are completely correct.

The human and Gemini evaluations indicate that factual support in the explanations is weaker than the automatic checking results suggest.

### Evaluation Charts

The evaluation charts are available in:

```text
results/figures/
```

They are labelled E1 to E7. The numerical evaluation summary is available in:

```text
results/evaluation_summary.txt
```

---

## Limitations

- **Limited automatic checking:** The automatic checker does not detect every incorrect or unsupported statement. Gemini identified 69 answers with unsupported information, while the checker flagged only two of them. Both checker-flagged answers were also identified by Gemini.
- **Explanation quality:** Based on the human and Gemini reviews, approximately 77-80% of explanations use only the facts provided.
- **On-ramp situations:** These scenarios were the weakest area for factual explanations.
- **Rule-based restrictions:** Rules A-D are easier to pass because the safety filter and required answer format already restrict the available actions.
- **Gemini limitations:** Gemini was stricter than the human reviewer and can also make mistakes. Agreement on the same 20 answers ranged from 75% to 85%.
- **Dataset limitations:** The project uses one motorway, NGSIM US-101, recorded on one morning in 2005. It has not been tested on UAE roads or under other traffic conditions.
- **Model performance:** A locally hosted 8-billion-parameter model is relatively slow, and answers can differ slightly between runs even with a fixed seed.
- **No real vehicle testing:** This is a decision-support prototype, not a complete self-driving system. It is not intended for use in a real vehicle.

---

## Dataset

The project uses the **NGSIM US-101 vehicle trajectory dataset**, provided by the U.S. Federal Highway Administration.

The data was recorded on 15 June 2005 on the Hollywood Freeway in Los Angeles, with vehicle positions and movements recorded at 10 readings per second.

The extracted dataset contains approximately:

- 4.8 million rows.
- 2,847 vehicles.

The original raw dataset (approximately 2 GB) is not included in this repository.

Instructions for downloading it are available in:

```text
data/README.md
```

The 300 prepared scenarios are already included in:

```text
ngsim_scenarios.jsonl
```

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language and system pipeline |
| Ollama | Runs the language model locally |
| Llama 3.1 8B | Selects and explains driving actions |
| Pandas | Traffic data processing |
| NumPy | Numerical calculations |
| Matplotlib | Evaluation charts |
| HTML and JavaScript | Web prototype interface |
| JSON / JSONL | Data storage and model responses |
| Gemini | Independent evaluation of model answers |
| GitHub | Source code and project version control |

---

## Project Status and Future Work

The junior project prototype and evaluation are complete.

Planned improvements include:

1. Developing a stronger explanation checker to identify more unsupported statements.
2. Involving a second human reviewer and reviewing more answers.
3. Replaying recommended actions on recorded traffic to evaluate their effects.
4. Exploring model fine-tuning and adaptation to UAE road conditions, in discussion with the supervisors.

The senior project will extend this work towards a small AI-powered mobile robot.

---

**University of Sharjah | Computer Science | Junior Project**
