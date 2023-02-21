import glob, os, random
import numpy as np


# データを訓練/評価/テストに分割する関数
# 参考サイト: https://qiita.com/QUANON/items/e28335fa0e9f553d6ab1
def divide(tmpList):
    random.shuffle(tmpList)
    arrayList = np.array(tmpList)

    # 6:2:2に分割
    divider = [int(arrayList.size*i) for i in [0.8, 0.8 + 0.1]]
    train, val, test = np.split(arrayList, divider)
    
    trainList.append(train.tolist())
    valList.append(val.tolist())
    testList.append(test.tolist())


# リストに格納したファイルパスを.txtに書き出す関数
# 参考サイト: https://www.mathpython.com/ja/python-file-write/
def list2txt(dataType, tmpList):
    with open(f"{DIR}/{dataType}.txt", "w") as f:
        f.write("\n".join(tmpList))


# main
DIR      = "NormalsleepFANZHANG"
fileType = "png"

trainList, valList, testList = [], [], []
fileList     = []
tmp_category = []

for categoryPath in glob.glob(f"{DIR}/*"):
    category = categoryPath.split(os.sep)[1]

    if tmp_category != category:
        tmp_category = category
        divide(fileList)
        if fileList: fileList = []
    
    for filePath in glob.glob(f"{DIR}/{category}/*.{fileType}"):
        file = filePath.split(os.sep)[1]
        fileList.append(f"{category}/{file}")

divide(fileList)

trainList = [i for row in trainList for i in row]
valList   = [i for row in valList   for i in row]
testList  = [i for row in testList  for i in row]


list2txt("train", trainList)
list2txt("val", valList)
list2txt("test", testList)