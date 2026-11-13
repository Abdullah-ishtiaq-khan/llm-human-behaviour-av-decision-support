# LLM-Based Autonomous Vehicle Decision Support System

## Project Overview

This project develops an AI-based decision support system for autonomous vehicles using real-world NGSIM traffic data and a Large Language Model (LLM).

The system analyses driving scenarios, applies a safety filter, generates driving recommendations using Llama through Ollama, and checks the recommendations for consistency and safety-related contradictions.

## Main Features

- 300 selected NGSIM driving scenarios.
- Scenario-based driving situation analysis.
- Custom driving scenario inputs.
- Safety filtering of available driving actions.
- Llama-based driving recommendations.
- Automatic consistency checking.
- Retry mechanism when an answer fails validation.
- Web-based interface for testing recommendations.

## Project Structure

- `prepare_us101.py` – Prepares the original NGSIM dataset.
- `ngsim_to_json.py` – Converts traffic data into driving scenarios.
- `run_pilot_ngsim_v5.py` – Runs the main recommendation pipeline.
- `check_answers.py` – Checks recommendation outputs.
- `server.py` – Python backend and API.
- `index.html` – Web interface.
- `ngsim_scenarios.jsonl` – Selected 300 driving scenarios.
- `results/` – Experiment results and review reports.
- `archive/` – Previous versions and historical files.
- `data/` – Dataset information.

## Technologies Used

- Python
- Pandas
- NumPy
- Ollama
- Llama
- HTML
- JavaScript
- JSONL
- NGSIM traffic dataset

## Running the Project

1. Install the required Python packages using `requirements.txt`.
2. Install and start Ollama with the required Llama model.
3. Run `python server.py` from the project folder.
4. Open `http://localhost:8000` in a browser.

## Dataset

The project uses the US-101 NGSIM vehicle trajectory dataset.

The raw dataset is not included because of its large size. Refer to `data/README.md` for dataset information.

## Project Status

The prototype includes a Python backend, a web interface, a scenario dataset, an LLM recommendation pipeline, and automatic output checking.

Further testing and evaluation are ongoing.