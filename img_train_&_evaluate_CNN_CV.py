import time
time_sta = time.time()

from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras import regularizers
from keras.applications.xception import Xception
from keras.applications.resnet import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
from keras.applications.vgg16 import VGG16

from keras.models import load_model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
# from sklearn.model_selection import KFold
# from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm
from PIL import Image
import numpy as np
import csv

print("\nGPU確認:")
# from client import device_lib
# print(device_lib.list_local_devices(),"\n")


## パラメータの設定
DATA_DIR = "0118"
# CLASSES  = [
#     "Flick_Down",
#     "Flick_Left",
#     "Flick_Right",
#     "Flick_Up",
#     "Grip",
#     "Palm_Down",
#     "Palm_Up",
#     "Push",
# ]
CLASSES = ["rightward", "leftward", "liedown", "situp"]
BATCH_SIZE     = 64
EPOCHS         = 100
cv_num         = 10      # 交差検証における積層数
width    = 50     # 画像の正方サイズ
height   = 37
early_stop_n   = 5      # 学習改善が見られない時のための停止猶予エポック数
debug          = False  # 交差検証を1回目で停止
augmentation   = False  # データ拡張の使用


# 画像処理 (オリジナルサイズ: 496 x 369)
def Proc_Image(img_path):
    img           = Image.open(img_path)
    resized_img   = img.resize((width, height))
    colored_img   = resized_img.convert("RGB")
    processed_img = np.asarray(colored_img)/255.
    w, h          = resized_img.size
    return processed_img, w, h

# モデルのアーキテクチャ
def Architecture():
    model = keras.Sequential([
        Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=(h, w, 3)),
        MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding="same",),
        MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(256,(3, 3), activation='relu', padding="same",),
        MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(CLASSES), activation='softmax')
    ])
    return model

# コールバック関数
def Callbacks(i):
    tb = TensorBoard(log_dir=f"./logs/{DATA_DIR}/{int(time_sta)}_{i+1}")
    #es = EarlyStopping(monitor='val_loss', patience=n)
    md = ModelCheckpoint(
        filepath=f"./checkpoints/{DATA_DIR}/{int(time_sta)}_{i+1}.h5",
        monitor="val_accuracy",  # val_accuracy   val_loss
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        period=1,
    )
    return [tb, md]


## 10分割交差検証（テストデータ）
for i in range(cv_num):

    # データの読み込みと前加工
    data = {}
    for mode in ["train", "val"]:
        x, y = [], []
        with open(f"data/{DATA_DIR}/{mode}_{i+1}.txt") as f:
            filenames = f.read().split("\n")
        
        for filename in tqdm(filenames, desc=mode):
            input_img, w, h = Proc_Image(f"data/{DATA_DIR}/{filename}")
            x.append(input_img)
            y.append(CLASSES.index(filename.split("/")[0]))
        
        x, y = np.array(x), np.eye(len(CLASSES))[y]
        data[mode] = {"x": x, "y": y}

    # データ拡張
    if augmentation:
        dataGen = ImageDataGenerator(
            featurewise_center            = False,      # データセット全体で入力の平均を0にする
            samplewise_center             = False,      # 各サンプルの平均を0にする
            featurewise_std_normalization = False,      # 入力をデータセットの標準偏差で正規化する
            samplewise_std_normalization  = False,      # 各入力をその標準偏差で正規化する
            zca_whitening                 = False,      # ZCA白色化の適用

            rotation_range                = 180.,       # 画像をランダムに回転する回転範囲
            width_shift_range             = 0.2,        # ランダムに水平シフトする範囲
            height_shift_range            = 0.2,        # ランダムに垂直シフトする範囲
            shear_range                   = 0.2,        # シアー変換(反時計回りのシアー角度)
            zoom_range                    = 0.2,        # ランダムにズームする範囲
            channel_shift_range           = 0.,         # ランダムにチャンネルをシフトする範囲
            fill_mode                     = "nearest",  # 入力画像の境界周り: {"constant", "nearest", "reflect", "wrap"}
            cval                          = 0.,         # fill_mode = "constant"のときに境界周辺で利用される値
            
            horizontal_flip               = True,       # 水平方向に入力をランダムに反転する
            vertical_flip                 = False,      # 垂直方向に入力をランダムに反転する
            rescale                       = None,       # 画素値のリスケーリング係数 
        )
        dataGen.fit(x)

    # モデルの構造の定義
    model = Architecture()
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # 学習の実行
    if augmentation:
        model.fit_generator(
            generator=dataGen.flow(x=data["train"]["x"], y=data["train"]["y"], batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=(data["val"]["x"], data["val"]["y"]),
            callbacks=Callbacks(i))
    else:
        model.fit(
            x=data["train"]["x"], y=data["train"]["y"],
            batch_size=BATCH_SIZE, epochs=EPOCHS,
            validation_data=(data["val"]["x"], data["val"]["y"]),
            callbacks=Callbacks(i))

    # モデルの保存
    model.save(f"models/{DATA_DIR}/model_{int(time_sta)}_{i+1}")


    ## テストデータを推論
    #model = load_model(f"models/{DATA_DIR}/model_{int(time_sta)}_{i+1}")
    model = load_model(f"checkpoints/{DATA_DIR}/{int(time_sta)}_{i+1}.h5")

    with open(f"evaluate/{DATA_DIR}/{int(time_sta)}_{i+1}.csv", mode="w", newline="") as f1:
        csv_writer = csv.writer(f1)
        csv_writer.writerow(["filename", *CLASSES, "Groundtruth", "Result",])

        with open(f"data/{DATA_DIR}/test_{i+1}.txt") as f2:
            filenames = f2.read().split("\n")

        for filename in tqdm(filenames):
            eval_img, w, h  = Proc_Image(f"data/{DATA_DIR}/{filename}")
            prediction      = model.predict(eval_img.reshape(1, h, w, 3))[0]
            result          = np.where(prediction == np.max(prediction))[0][0]
            csv_writer.writerow([filename, *prediction, filename.split("/")[0], CLASSES[result]],)
    
    if debug: break


## 時間計測
time_end   = time.time()
tim        = time_end- time_sta
tim_hour   = int(tim // 3600)
tim_minute = int((tim % 3600) // 60)
tim_second = int((tim % 3600 % 60))
print(str(tim_hour).zfill(2) + "h " + str(tim_minute).zfill(2) + "m " + str(tim_second).zfill(2) + "s")
