# Boston-Home-Price-Forecast
# 一、說明
本專題透過Kaggle用波士頓房價預測比賽，用監督使學習方式訓練資料集與測試集之線性迴歸、PCA降維訓練出模型個關係定義，並把這個關係用測試的資料做驗證，確認我們找出的房子特徵是否能夠準確預測房價。
# 二、實作(請使用python3.6版做執行)
透過 Kaggle 資料競賽網站，下載波士頓房價資料集。(Link: https://www.kaggle.com/competitions/boston-housing/data)

(點擊 "Download All" 後解壓縮，並透過下方程式碼上傳 submission_example.csv, test.csv , train.csv 三份檔案)

※ 可一次上傳或分批上傳，上傳成功後，點擊左方 "Files" 欄位，即可看到上傳的檔案。

首先透過Kaggle資料競賽網站，下載波士頓房價資料集。使用pandas 匯入訓練集與測試集資料，並利用線性迴歸、PCA降維訓練出模型。以下是預測房價中文資料集。
![image](https://github.com/LonelyCaesar/-Boston-Home-Price-Forecast/assets/101235367/a1d49353-6730-453f-9fbf-7dc9e0e5f2b8)
### 1.	讀取資料合併：
使用sklearn提供的數據，來做分析，如此一來就不用再引入csv檔，而我們用來舉例的是load_boston()，接下來就讓我們一步步的來瞭解資料，使用分析的結果，獲得資訊吧。
### 程式碼：
```python
# 忽略警告訊息
import warnings
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import datasets
boston_dataset = load_boston()
print(boston_dataset.DESCR)
data = datasets.load_boston()

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
submit = pd.read_csv('submission_example.csv')

print('train',df_train.shape)
display(df_train.head(5))
print('test',df_test.shape)
display(df_test.head(5))

# 合併train及test的資料 
df_data = df_train.append(df_test)
df_data
```
