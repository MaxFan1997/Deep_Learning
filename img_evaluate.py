import csv
import time

import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm
from PIL import Image


DATA_DIR = "0123"
# CLASSES = [
#     "LP",
#     "LS",
#     "PL",
#     "PR",
#     "RP",
#     "RS",
#     "SL",
#     "SR"

# ]
# CLASSES = ["rightward", "leftward", "liedown", "situp"]
CLASSES = ["sitting","turning"]
model = load_model("models/model_1674460491")

# CSVの書き込む準備


timestamp = int(time.time())
with open(f"evaluate/{timestamp}.csv", mode="w", newline="") as f1:
    csv_writer = csv.writer(f1)
    csv_writer.writerow([
        "filename",
        *CLASSES
    ])

    # test.txtからファイル名を読み込む
    with open(f"{DATA_DIR}/test.txt") as f2:
        filenames = f2.read().split("\n")

    for filename in tqdm(filenames):
        # CSVファイルを読み込んで推論する
        img = Image.open(f"{DATA_DIR}/{filename}")
        img = img.resize([100,74])
        img = img.convert("RGB")

        x = np.asarray(img)

        prediction = model.predict(x.reshape(1, 74,100, 3))[0]

        # 推論した結果をCSVに書き込む
        csv_writer.writerow([
            filename,
            *prediction
        ])
