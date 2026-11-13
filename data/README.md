# NGSIM Dataset

This project uses the US-101 NGSIM vehicle trajectory dataset.

The original dataset is not included in this repository because of its large file size.

Download the dataset from the official US Department of Transportation source:

https://data.transportation.gov/

The raw dataset is processed using `prepare_us101.py` to create `us101_clean.csv`.

The cleaned dataset is then converted into driving scenarios using `ngsim_to_json.py`.

The generated `ngsim_scenarios.jsonl` file contains 300 selected driving scenarios used by this project.