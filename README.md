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
### 執行結果：
![image](https://github.com/LonelyCaesar/-Boston-Home-Price-Forecast/assets/101235367/4ecb9a67-ce6f-4f68-a653-f0e3c7233247)

合併後訓練測試集總共有506筆資料、15個特徵欄位，做出一致性的預測分析及模型訓練就會比較快速好理解。
### 2.線性迴歸
首先，要使用簡單的資料視覺來看一下細部資料之間的關係，用MEDV房價變數做分佈的線性迴歸常態預測。
### 程式碼：
```python
boston = pd.DataFrame(boston_dataset.data,
                     columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
sns.distplot(boston['MEDV'], bins=30)
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Boston-Home-Price-Forecast/assets/101235367/72c9485b-4cfd-4ca0-bf77-4344c20315da)

接者我們可以看每個變數之間的關係，透過相關係數觀察特徵變數和目標變數有較高的關聯性。
### 程式碼：
```python
#使用熱度圖產生模型圖
correlation_matrix = boston.corr().round(2)
plt.figure(figsize=(15,9))
sns.heatmap(correlation_matrix, annot=True)
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Boston-Home-Price-Forecast/assets/101235367/ebfbf89d-dcac-4cb5-a474-7adeaa6bda2e)

使用LSTAT和RM來做出預測MEDV的模型。用下列的算數及預測圖將關係數值給分析出來，可以明顯看到兩者之間的關係會是怎麼樣。
### 程式碼：
```python
X = boston.loc[:,"RM":"LSTAT"].values
Y = boston.MEDV
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=9487)
regr = LinearRegression()
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("MAE =","%.4f" % mae)
print("MSE =","%.4f" % mse)
print("R2 =","%.4f" % r2)
```
### 執行結果：
列印出來的模型的平均絕對物誤差3.5028、均方誤差26.4660、判定係數為得到的平均於71%，代表它的分析程度是相當好的。
### 程式碼：
```python
# 設定整張圖的長寬
plt.figure(figsize=(20, 5))
features = ["RM","LSTAT"]
target = boston['MEDV']
for i, col in enumerate(features):
# 排版1 row, 2 columns, nth plot：在jupyter notebook上兩張並排 
 plt.subplot(1, len(features) , i+1)
 # add data column into plot
 x = boston[col]
 y = target
 plt.scatter(x, y, marker='o')
 plt.title(col)
 plt.xlabel(col)
 plt.ylabel('MEDV')
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Boston-Home-Price-Forecast/assets/101235367/c42e70d6-7e0b-4eb2-9f84-697382a6467a)

左圖：(RM與MEDV)住宅的平均房間數與房子的中位數價格產生了正向關係，也就是說平數越多價錢就會變高，看個人需求而自行決定。 右圖：(LSTAT與MEDV)人口數量與房子的中位數價格產生了負向關係，人口大於房子數量就會影響了遮風避雨無家可歸的現象發生。
### 3.	產生模型的重要性：
使用13個特徵產生出多張的模型，此模型會跟上述熱度圖很類似，因為執行要一點時間就挑選前幾個特徵做執行，再讀取前8筆的資料訓練出來的模型去預測。
### 程式碼：
```python
sns.set()
cols = ['crim','zn','indus','chas','nox','rm','age','dis']
sns.pairplot(df_data[cols], size = 2.5)
plt.show()
```
### 執行結果：
執行結果為特徵圖與直條圖，與上述熱度圖的範疇。
![download](https://github.com/LonelyCaesar/-Boston-Home-Price-Forecast/assets/101235367/18ffbc4b-efde-4966-9d53-964f6c41f058)

### 程式碼：
```python
boston = load_boston()
df = pd.DataFrame(data=boston.data, 
                  columns=boston.feature_names)
df['MEDV'] = boston.target

# MEDV即預測目標向量
X = df.iloc[:, :-1]
y = df['MEDV']
X.head(5)
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Boston-Home-Price-Forecast/assets/101235367/b20a9fce-d749-43ef-88d4-aa971c3161a6)

將13特徵合併成一張斜線圖與點點圖。
### 程式碼：
```python
x = data.data # 13個特徵的數據
y = data.target # 房價數據

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1) # 將數據分成73比
lnregr = LinearRegression()
lnregr.fit(x_train, y_train) # 將資料拿去訓練

y_predict = lnregr.predict(x_test) # 北test的資料用訓練出來的模型去預測

plt.xlabel("actual price") # x軸的標題
plt.ylabel("predict pcice") # y軸的標題
plt.plot([0,50], [0,50]) # 劃一條基準線
plt.scatter(y_test, y_predict) # 比對預測跟實際的差別
plt.show() # 察看結果
lnregr.score(x_train, y_train)
```
### 執行結果：
為0.7103879080674731
![image](https://github.com/LonelyCaesar/-Boston-Home-Price-Forecast/assets/101235367/8c44e303-056a-475e-ba23-99308f37f961)

執行後訓練出來的特徵數值維71%，預測值機率只有7成，難以預測房價的特性狀況，換個其他資料做預測依然還是大同小異。
### 4.	PCA降維：
訓練測試集總共有506筆資料、15個特徵欄位，使用的數據量會不足，因此我們使用降維方式做出線性圖模型。
### 程式碼：
```python
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

x = data.data # pca降維後的數據
y = data.target # 房價數據

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1) # 將數據分成73比

# Standarize our training data
std_tool = StandardScaler()
std_tool.fit(x_train)
x_train = std_tool.transform(x_train)

# PC降維
pca = decomposition.PCA(n_components=0.95)
pca.fit(x_train)
x_train = pca.transform(x_train)
lnregr = LinearRegression()
lnregr.fit(x_train, y_train) # 將資料拿去訓練

# Standarize x_test
x_test = std_tool.transform(x_test)

# Dimension reduction usng PCA
x_test = pca.transform(x_test)
y_predict = lnregr.predict(x_test) # 將test的資料用訓練出來的模型去預測

plt.xlabel("actual price") # x軸的標題
plt.ylabel("predict pcice") # y軸的標題
plt.plot([0,50], [0,50]) # 劃一條基準線
plt.scatter(y_test, y_predict) # 比對預測跟實際的差別
plt.show() # 察看結果
lnregr.score(x_train, y_train) # 訓練模型的正確率
```
### 執行結果：
為0.6649582793264731
![image](https://github.com/LonelyCaesar/-Boston-Home-Price-Forecast/assets/101235367/b430f20f-5891-4310-85f5-a3c9c658ea8b)

PCA降維所訓練出來得到的為67%，由此可知低，於了上述的訓練出的模型，解釋程度能說是相當好的。
### 程式碼：
```python
submit.to_csv( 'New_SampleSubmission1.csv', index=False )
print( f'預測結果：' )
submit.head(20)
```
產生後儲存為資料表後上傳至Kaggle的Submit Predictions，然後按Submit就完成了此競賽項目。
# 三、結論
使用 Kaggle 上波士頓房價預測比賽，使用了房子數量、人口數量做出相關關資料的分析及處理技巧，也用線性迴歸預測、產生模型的重要性及降維來預測、觀察及嘗試。畢竟房價不像用斤兩來秤有絕對的標準，更何況房價還要依賣方的心情，當前的通膨，人均所得，根本就沒有標準。人量不出來的，更別想叫電腦算的準。不過，我們自已調整及建立的模型，將 score 調到最高，雖說還是不準確，但絕對是值得參考的依據。
# 四、參考
Hahow學習AI一把抓：點亮人工智慧技能樹。

Hahow Python資料分析：AI機器學習入門到應用。
