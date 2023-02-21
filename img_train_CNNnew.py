import time
time_sta = time.time()

# from tensorflow import keras
# from tensorflow.python.keras.layers import Conv1D, Conv2D, MaxPool1D, MaxPool2D, Flatten, Dense, Dropout
# from tensorflow.python.keras.callbacks import TensorBoard
# from tqdm import tqdm
# from PIL import Image
# import numpy as np

from tensorflow import keras
from keras.layers import Conv1D, Conv2D, MaxPool1D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import TensorBoard
from tqdm import tqdm
from PIL import Image
import numpy as np

DATA_DIR = "NormalsleepFANZHANG"

# CLASSES = [
#     "Flick_Down",
#     "Flick_Left",
#     "Flick_Right",
#     "Flick_Up",
#     "Grip",
#     "Palm_Down",
#     "Palm_Up",
#     "Push",
# ]
CLASSES = ["Prone", "Supine", "Left", "Right", "Sit", "Stand"]
# CLASSES = ["sitting","turning"]
BATCH_SIZE = 512
EPOCHS = 100






# モデルの構造の定義
model = keras.Sequential([
    Conv2D(128, (5, 5), activation="relu", padding="same", input_shape=(37, 50, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(256, (5, 5), activation="relu", padding="same"),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Dropout(0.4),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.4),
    Dense(128, activation="relu"),
    Dropout(0.4),
    # BatchNormalization(),
    Dense(len(CLASSES), activation="softmax"),
])
model.compile(optimizer="Adam", # "Adam", keras.optimizers.Adam(lr=0.0001)
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# データの読み込みと前加工
data = {}
for mode in ["train", "val"]:
    with open(f"{DATA_DIR}/{mode}.txt") as f:
        filenames = f.read().split("\n")
    x = []
    y = []
    for filename in tqdm(filenames, desc=mode):
        # 画像データの入力
        img = Image.open(f"{DATA_DIR}/{filename}")
        img = img.resize((50, 37)) # 画像サイズの変更
        img = img.convert("RGB")

        x.append(np.asarray(img))
        y.append(CLASSES.index(filename.split("/")[0]))
    x = np.array(x)
    y = np.eye(len(CLASSES))[y]
    data[mode] = {"x": x, "y": y}

# 学習の実行
timestamp = int(time.time())
model.fit(x=data["train"]["x"],
          y=data["train"]["y"],
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(data["val"]["x"],
                           data["val"]["y"]),
          callbacks=[TensorBoard(log_dir=f"./logs/{timestamp}")])

# モデルの保存
model.save(f"models/model_{timestamp}")

time_end = time.time()
tim = time_end- time_sta
print(tim)