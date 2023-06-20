#ライブラリのインポート
import numpy as np
import pandas as pd
import csv

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#データの読み込み
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#学習データとテストデータを連結
df = pd.concat([train_df,test_df],ignore_index = True)
tmp = df.groupby('Sex').agg({'Sex':'count'}).rename(columns = {'Sex':'count_Sex'})

#元データをコピー
df2 = df.copy()
#欠損値の補完(Embarked)
df2.Embarked = df2.Embarked.fillna('S')

#元データをコピー
df3 = df2.copy()
#欠損値の補完(Age)
age_median = df3.Age.median()
df3.Age = df3.Age.fillna(age_median)

#今回使わないカラム(特徴量)を削除する
df4 = df3.drop(columns = ['Name','SibSp','Parch','Ticket','Fare','Cabin'])

#ワンホットエンコーディング(Embarked)
tmp_Embarked = pd.get_dummies(df4['Embarked'],prefix = 'Embarked')
df5 = pd.concat([df4,tmp_Embarked],axis = 1).drop(columns = 'Embarked')

#ワンホットエンコーディング(性別)
df5['Sex'] = pd.get_dummies(df5['Sex'],drop_first=True)

#学習データに分割した結果を変数trainに格納する
train = df5[~df5.Survived.isnull()]

#テストデータに分割した結果を変数testに格納する
test = df5[df5.Survived.isnull()]

#Survivedを削除
test = test.drop(columns=['Survived'])

#正解をy_trainに格納する
y_train = train.Survived

#特徴量をX_trainに格納する
X_train = train.drop(columns = ['Survived'])

#決定木モデルの準備
from sklearn import tree
model = tree.DecisionTreeClassifier()

#決定木モデルの作成
model.fit(X_train,y_train)

y_pred = model.predict(test)
test['Survived'] = y_pred

#提出用のデータマートを作成する
pred_df = test[['PassengerId','Survived']].set_index('PassengerId')

#予測結果を整数に変換する
pred_df.Survived = pred_df.Survived.astype(int)
print(pred_df.head())

#CSVの作成
pred_df.to_csv('submission_v1.csv',index_label= ['PassengerId'])


