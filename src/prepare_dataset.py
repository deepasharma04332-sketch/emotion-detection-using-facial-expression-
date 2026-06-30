"""
prepare_dataset.py
Helper script to organize the FER-2013 dataset (from Kaggle, CSV format)
into the train/test folder structure expected by train.py.

Some Kaggle versions of FER-2013 come as a single fer2013.csv file with
columns: emotion, pixels, Usage. This script converts that CSV into
labeled image folders.

Steps to get the dataset:
    1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
       (or search "FER2013" on Kaggle)
    2. Download and unzip it.
    3a. If you get a CSV file (fer2013.csv):
        - Place it inside the data/ folder as data/fer2013.csv
        - Run: python src/prepare_dataset.py
    3b. If you get folders already split into train/test with emotion
        subfolders, just place them directly as:
            data/train/Angry, data/train/Happy, ...
            data/test/Angry, data/test/Happy, ...
        and skip this script.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "fer2013.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")

EMOTION_MAP = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


def main():
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found at {CSV_PATH}.")
        print("Download fer2013.csv from Kaggle and place it in the data/ folder.")
        return

    print("Reading CSV... this may take a minute.")
    df = pd.read_csv(CSV_PATH)

    for _, row in df.iterrows():
        emotion_label = EMOTION_MAP[int(row["emotion"])]
        usage = row["Usage"]  # "Training" or "PublicTest"/"PrivateTest"
        split = "train" if usage == "Training" else "test"

        pixels = np.array(row["pixels"].split(), dtype="uint8").reshape(48, 48)
        img = Image.fromarray(pixels, mode="L")

        out_dir = os.path.join(OUTPUT_DIR, split, emotion_label)
        os.makedirs(out_dir, exist_ok=True)

        img_path = os.path.join(out_dir, f"{_}.png")
        img.save(img_path)

    print("Dataset successfully converted into data/train and data/test folders.")


if __name__ == "__main__":
    main()
