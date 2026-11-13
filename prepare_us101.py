"""
prepare_us101.py - step 1 of the NGSIM project.

The NGSIM file is big (about 2 GB) and holds four sites. This script reads it in chunks
(so it does not fill your RAM), keeps only the US-101 rows and the columns we need,
and saves them to a much smaller file: us101_clean.csv

Run in the folder that has ngsim_data.csv:
    python prepare_us101.py
Needs: pandas (pip install pandas)
"""
import sys
import pandas as pd

INPUT = sys.argv[1] if len(sys.argv) > 1 else "ngsim_data.csv"
OUTPUT = "us101_clean.csv"
CHUNK = 500_000

KEEP = ["Vehicle_ID", "Frame_ID", "Global_Time", "Local_X", "Local_Y", "v_length", "v_Width",
        "v_Class", "v_Vel", "v_Acc", "Lane_ID", "Preceding", "Following",
        "Space_Headway", "Time_Headway", "Location"]


def norm(s):
    return str(s).strip().lower().replace("-", "").replace("_", "").replace(" ", "")


print("Reading", INPUT, "in chunks of", f"{CHUNK:,}", "rows ...")
counts = {}
kept = 0
first = True
for i, chunk in enumerate(pd.read_csv(INPUT, usecols=KEEP, chunksize=CHUNK, thousands=",")):
    for loc, n in chunk["Location"].value_counts().items():
        counts[loc] = counts.get(loc, 0) + int(n)
    sub = chunk[chunk["Location"].map(norm) == "us101"]
    if len(sub):
        sub.to_csv(OUTPUT, mode="w" if first else "a", header=first, index=False)
        first = False
        kept += len(sub)
    print(f"  chunk {i + 1}: total rows so far {sum(counts.values()):,} | US-101 rows kept {kept:,}")

print("\nRows per Location value in the file:")
for k, v in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {k!r}: {v:,}")

if kept == 0:
    print("\nNo US-101 rows found. Look at the Location values above and tell me what they are.")
    sys.exit(1)

df = pd.read_csv(OUTPUT)
print(f"\nSaved {OUTPUT}: {len(df):,} rows, {df['Vehicle_ID'].nunique():,} vehicles")
print("Lanes:", sorted(df["Lane_ID"].unique().tolist()))
print("Vehicle classes (1=motorcycle, 2=car, 3=truck):", df["v_Class"].value_counts().to_dict())
print("Speed (ft/s) min/mean/max:", round(df["v_Vel"].min(), 1), round(df["v_Vel"].mean(), 1), round(df["v_Vel"].max(), 1))
print("Accel (ft/s2) min/max:", round(df["v_Acc"].min(), 1), round(df["v_Acc"].max(), 1))
print("Rows with a vehicle ahead (Preceding > 0):", f"{(df['Preceding'] > 0).sum():,}")
print("Frames per second: 10. Time span (min):", round((df["Global_Time"].max() - df["Global_Time"].min()) / 60000, 1))
