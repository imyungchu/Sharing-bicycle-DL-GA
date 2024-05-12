
結合需求預測之共享單車再配送專題研究
Sharing bicycle demand forecasting and redistribution Routing Project Research
===
> 1. Project Full Name：
> 
> a. 中文：結合深度學習預測站點供需量之共享單車再配送路徑問題
> 
> b. English：Sharing Bicycle Redistribution Routing Problem Integrating Station Demand Forecasting Using Deep Learning
> 
> 2. The period of this project：2021.7~2022.6
> 
> 3. Advising Professor：[Chun-ChengLin](https://cclin321.iem.nycu.edu.tw/chun-cheng-lin)
> 
> 4. Project Member：Chin-Chuan Cheng、Huei-Yu Hsieh、Yu Yen、[Yung-chu Chiang*](http://imyungchu.github.io)
> 
> 5. Ｍy Contribution：
> 
> a. Representitive and communicator
> 
> b. Write down most of the papers and plans
> 
> c. Work on PPT and video presentation 
> 
> 🏆 Honorable Mention (Big Data and Information Systems),
> 2022 CIIE Student Monograph Competition
> 大學生專題論文佳作(大數據分析與資訊系統組)，
> 2022全國工業工程與管理大學生專題論文與技術報告競賽
> 
> 🏆 Honorable Mention Award
> 2022 NYCU 2nd Sustainable Development Goals Student Research Poster Awards
> 佳作，2022 國立陽明交通大學第二屆聯合國永續發展目標SDGs論文海報競賽
> 
> Copyright© No one shall download, reprint, reproduce, distribute, publicly broadcast, publish or distribute the research content of our project research in any form without written consent.©
> 
## Quick Guide

If you want to quickly go through this project, start here!

1. Watch our [Youtube presentation video ](https://www.youtube.com/watch?v=NEtkb0T8vGE)
2. Read our full paper ([ chinese ](https://drive.google.com/file/d/1kp41gxX0jrnYMOhmS2N_PKk_ILE-XPiQ/view?usp=sharing)/ english )
3. View our project code ( [Demand forcasting ](https://drive.google.com/drive/folders/1V76LZZrjoxJZziA-NIosvwE0mHmeKjaA?usp=sharing)/ [Sharing bicycle redistribution](https://) )
4. [Poster Link](https://sdgs.nycu.edu.tw/wp-content/uploads/017-%E4%BD%B3%E4%BD%9C-%E7%B5%90%E5%90%88%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E9%A0%90%E6%B8%AC%E7%AB%99%E9%BB%9E%E4%BE%9B%E9%9C%80%E9%87%8F%E4%B9%8B%E5%85%B1%E4%BA%AB%E5%96%AE%E8%BB%8A%E5%86%8D%E9%85%8D%E9%80%81%E8%B7%AF%E5%BE%91%E5%95%8F%E9%A1%8C.pdf)

Abstrct
---
###  1. Motivation of this project ? Why few sharing bicycle system can success ?
The rising wave of the sharing economy in the world has prompted a large amount of capital and technology to be invested in the sharing bicycle system. Because bicycle sharing is **environmentally friendly** and can prevent traffic congestion, sharing bicycles are an effective choice to replace traditional high carbon emission transportation. In addition, sharing bicycles can **reduce the number of any possible risk of social contact** that mainly occur in public places, and this is especially important in today's severe COVID-19 pandemic. 

Under the **uncertainty of demand**, sharing bicycles system are prone to encounter **supply and demand unbalanced**, and the system also **lacks a good bicycle redistribution strategy**. As a result, there is almost no successful sharing bicycle system, and it has attracted many people ,including us, work on this topic.


### 2. What are the differents between our project and others?
Most of the previous studies considered simply station Demand Forecasting or sharing bicycle redistribution routing.Only considering the former is unable to actually adjust the number of sharing bicycles at the stations. While only consider the latter can optimize the route according to historical data, it cannot effectively cope with the changing demand for bicycles in real time and space. Therefore, this study **integrating Sharing Bicycle Redistribution Routing Problem with Station Demand Forecasting**. 

#### a. Studying methods
The demand forecasting of sharing bicycle stations is a **time series forecasting problem**, and sharing bicycle redistribution routing is a kind of **Vehicle Routing Problem with Pickup and Delivery (VRPPD)** which are NP-hard complex problems. Our study further proposes a **long short-term memory (LSTM)** and **Gated Recurrent Unit (GRU)** ,and they are all Recurrent Neural Network (RNN) models of deep learning. 

#### b. Studying procedures
Use the methods and models above to establish three prediction models to predict the demand for sharing bicycles at each station hourly in the future.And then the caculated and **predicted number of bicycles in and out of each station** is used as the input of the redistribution model to minimize carbon emissions and the total amount of redistribution distance.We establishes a Mathematical Programming for **reducing total distance and cost**, and finally uses the **Genetic Algorithm(GA)** to optimize sharing bicycle redistribution routing problem. 

### 3. How to implement and evaluate the idea of our project?
To evaluate the proposed method, this study work on some programming experience using the open dataset of [Capital Bikeshare](https://ride.capitalbikeshare.com/system-data), the largest bike-sharing system in the United States, located in Washington DC.  Quantitative historical data and attributes that affect demand are tested for a simulated experimental data set. 
#### Conclusion
The experiments confirm that our method proposed in this study, implementing with the **Keras** open source library, can **reduce 36% of the total distance** and carbon emissions, reaching the goal of energy saving, carbon reduction and **sustainable operation of the sharing bicycle system**.
```python=
# We use cross validation (Train data 70%,Testing data30%)
# The function cross validation

def build_training_and_validation_data(x_train, y_train, rate):
  X_train = x_train[int(x_train.shape[0]*rate):]
  Y_train = y_train[int(y_train.shape[0]*rate):]
  X_val = x_train[:int(x_train.shape[0]*rate)]
  Y_val = y_train[:int(y_train.shape[0]*rate)]
  return X_train, Y_train, X_val, Y_val
```
> Read more about our project study here:
> [Youtube presentation video ](https://www.youtube.com/watch?v=NEtkb0T8vGE), [ chinese full paper](https://drive.google.com/file/d/1kp41gxX0jrnYMOhmS2N_PKk_ILE-XPiQ/view?usp=sharing)

The code of our project
---
### Descriptioon 

The dataset of this project are downloaded from [Capital Bikeshare official website](https://ride.capitalbikeshare.com/system-data) (2018~2021)
There are two parts of our codes : 

1. Demand forcasting using Deep learning ( LSTM / GRU )
```python=
#Packages we used
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import gradient_descent_v2 
import matplotlib.pyplot as plt

#模型架構
#LSTM
def buildManyToOneModel(shape):
  model = Sequential()
  model.add(LSTM(20, input_shape=(shape[1],shape[2]),return_sequences=True))
  model.add(LSTM(10, input_shape=(shape[1],shape[2]),return_sequences=False))
  model.add(Dropout(0.05))
  model.add(Dense(10))
  model.add(Dense(1))
  model.compile(loss="mse", optimizer="Adam", metrics=['accuracy', 'mse'])
  model.summary()
  return model

#GRU
def buildManyToOneModel(shape):
  model = Sequential()
  model.add(GRU(25,input_shape=(shape[1],shape[2]),return_sequences=True))
  model.add(Dropout(0.2))
  model.add(GRU(20, input_shape=(shape[1],shape[2]),return_sequences=True))
  model.add(Dropout(0.2))
  model.add(GRU(15,input_shape=(shape[1],shape[2]),return_sequences=True))
  model.add(Dropout(0.2))
  model.add(GRU(units=50, activation='tanh'))
  model.add(Dropout(0.2))

  model.add(Dense(1))
  model.compile(loss="mse", optimizer="Adam")
  # model.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
  model.summary()
  return model
```

2. Optimizing Vehicle Routing Problem with Pickup and Delivery using Gnetic Algorithm
```python=
#fitness function
def fitFunc(x1, x2):
    vtotal = 0
    v_dist = np.zeros((NUM_VEHICLE,), dtype=float)
    pre_city = np.zeros((NUM_VEHICLE,), dtype=int)
    weight = np.full((NUM_VEHICLE,), INIT_WEIGHT)
    route = np.zeros([NUM_VEHICLE, NUM_POINT], dtype=int)
    
    for i in range(NUM_BIT):  # 站點分給每一台車
        j = np.argmin([math.hypot(location[x1[i]][0] - x2[2 * k], location[x1[i]][1] - x2[2 * k + 1]) for k in range(NUM_VEHICLE)])
        v_dist[j] += dist[pre_city[j]][x1[i]]
        pre_city[j] = x1[i]

        for a in range(NUM_POINT):      # 將站點記錄起來
            if route[j][a] == 0:
                route[j][a] = x1[i]
                break

    for i in range(NUM_VEHICLE):  # 加上回到起點的距離
        v_dist[i] += dist[pre_city[i]][0]

    for i in range(NUM_VEHICLE):
        for j in range(NUM_POINT):  #算即時載重
          if route[i][j] != 0:
            weight[i] = weight[i] + need[route[i][j] - 1]
            if weight[i] < 0 or weight[i] > MAX_WEIGHT:
              v_dist[i] = 100000
          else:
            break

    for i in range(NUM_VEHICLE):  # 算總距離
        vtotal += v_dist[i]


    return -(vtotal)  # 回傳總路徑長
```

> Read more about our code here:[DL Demand forcasting](https://drive.google.com/drive/folders/1V76LZZrjoxJZziA-NIosvwE0mHmeKjaA?usp=sharing), [VRPPD redistribution](https://drive.google.com/drive/folders/141R3ZnRBsyUL0MD5EX_0aLYUgw4p6V1g?usp=sharing)   



###### tags: `專題` `深度學習` `共享單車系統` `共享單車再配送路徑問題` `深度學習` `數學規劃` `基因演算法` `單車需求預測`
