# Stock_prediction with 3 perspective (2020.01 ~ 2020.08)

### Team member
JeongMan Choi(https://github.com/JeongmanChoi), SuYeon Jo(https://github.com/susie202404), SangHyung Jung

### Overview
Aim to predict stock up&down with Behavior View, Fundamental View, Technical View

## 1. Behavior view
Behavior analysis is a natural science that seeks to understand the behavior of individuals

### Neural Tensor Network
For Behavior view, We utilized NTN for event Embedding.

Before training, we preprocessed text data with keyword from naver news. For example, we select 'Swine fever' keyword for Livestock industry. You can check the code in text_preprocess.py

After preprocessing, we fine tuned bert model and get the sentence vector from bert model. You can check the code in fine_tuning/

Finally we made event embedding model through NTN and get the event vector. You can check the code in NTN/


## 2. Fundamental view
Fundamental analysis is a method of evaluating the intrinsic value of an asset and analysing the factors that could influence its price in the future. This form of analysis is based on external events and influences, as well as financial statements and industry trends.

We collect several data that represent the companies' financial statements and industrial factors from various routes such as Investing.com.

For example, we collect beans price for Livestock industry.


## 3. Technical View
Technical indicators are heuristic or pattern-based signals produced by the price, volume, and/or open interest of a security or contract used by traders who follow technical analysis.

We made lots of technical indicators that made up of open, close, volume, high, low. We utilized Ta-Lib package.


## 4. Portfolio
We used boosting model for all company such as LGBM, XGBOOST, CATBOOST.

We ensemble all these model and get a RandomSearch for optimal accuracy.

Finally, we built a portfolio from the results of each company's model.


## 5. Presentation
You can watch our presentation through the link below.

https://www.youtube.com/watch?v=94N4jp5LqYM&list=PLThNmt_l7b6CqH3cDSJQjMVBKQRuORwTW&index=6

## to do : fundamental & technical 변수들 daily update 되도록 수정 및 모듈화 필요
