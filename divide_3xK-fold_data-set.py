import glob, os, random
import numpy as np


# データを訓練/評価/テストに分割する関数
# 参考サイト: https://qiita.com/QUANON/items/e28335fa0e9f553d6ab1
def divide(tmpList):
    random.shuffle(tmpList)
    arrayList = np.array(tmpList)

    # 分割
    train, test = np.split(arrayList, [int(arrayList.size * 0.88888889)])
    return train, test


# リストに格納したファイルパスを.txtに書き出す関数
# 参考サイト: https://www.mathpython.com/ja/python-file-write/
def list2txt(dataType, tmpList):
    with open(f"{DIR}/{dataType}.txt", "w") as f:
        f.write("\n".join(tmpList))


# main
DIR          = "0118"
tmp_category = ""
trainList_i  = [[],[],[],[],[],[],[],[],[],[]]
valList_i    = [[],[],[],[],[],[],[],[],[],[]]
testList_i   = [[],[],[],[],[],[],[],[],[],[]]
count        = 0
length       = len(trainList_i)

for filePath in glob.glob(f"{DIR}/*/*.png"):
    category, fileName = filePath.split(os.sep)[1:]
    # gesture_num        = fileName.split("__")[1].split("-")[1]

    count += 1
    for i in range(length):
        if count % length != i:
            trainList_i[i].append(f"{category}/{fileName}")
    testList_i[count % length].append(f"{category}/{fileName}")


for i in range(length):
    trainList_i[i], valList_i[i] = divide(trainList_i[i])

    list2txt(f"train_{i+1}", trainList_i[i])
    list2txt(f"val_{i+1}", valList_i[i])
    list2txt(f"test_{i+1}", testList_i[i])