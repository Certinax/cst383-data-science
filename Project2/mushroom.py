#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:11:38 2019

@author: certinax
"""

import numpy as np
import pandas as pd
from multiprocessing import  Pool
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/ThoMot/DataHost/master/DataScience2019/mushrooms.csv")

#df.info()

#df.isna().sum()

#df["class"].value_counts()

df["class"] = (df["class"] == "e").astype(int)

df.set_index(df["class"], inplace=True)

df.head()

df.drop(columns=["class"], inplace=True)

cols = df.columns[df.columns != "stalk-root"]
cols

height = int(np.ceil(len(cols)/2))
print(height)



#df["ring-type"].value_counts().plot(ax=axes[0], kind='bar')
fig, axes = plt.subplots(height, 2)
#df["odor"].value_counts().plot(ax=axes[0], kind='bar')
#df["bruises"].value_counts().plot(ax=axes[1], kind='bar')
#df["cap-color"].value_counts().plot(ax=axes[2], kind='bar')
#df["ring-type"].value_counts().plot(ax=axes[3], kind='bar')

z = 0
for i in cols:
    df[i].value_counts().plot(ax=axes[z], kind='bar')
    z = z+1

print(z)



df2 = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
for i, col in enumerate(df2.columns):
    print(col)
    df2[col].plot(kind="box", ax=axes[i])



test = df.columns[:4]
fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True)
row = 0
for i, col in enumerate(test):
    print("Value of row {:.0f}".format(row))
    print("Value of i {:.0f}".format(i))
    #df[col].value_counts().plot(kind="bar", ax=axes[row,i])
    if(i != 0):
        #print("I ER IKKE NULL")
        if((i%2) != 0):
            row = row + 1
    sns.countplot(df[col], ax=ax[row,i])


df.info()
fig, ax = plt.subplots(10,2, figsize=(12,20))
sns.countplot(df["class"], ax=ax[0,0])
sns.countplot(df["cap-shape"], ax=ax[0,1])
sns.countplot(df["cap-surface"], ax=ax[1,0])
sns.countplot(df["cap-color"], ax=ax[1,1])
sns.countplot(df["bruises"], ax=ax[2,0])
sns.countplot(df["odor"], ax=ax[2,1])
sns.countplot(df["gill-attachment"], ax=ax[3,0])
sns.countplot(df["gill-spacing"], ax=ax[3,1])
sns.countplot(df["gill-size"], ax=ax[4,0])
sns.countplot(df["gill-color"], ax=ax[4,1])
sns.countplot(df["stalk-shape"], ax=ax[5,0])
sns.countplot(df["stalk-root"], ax=ax[5,1])
sns.countplot(df["stalk-surface-above-ring"], ax=ax[6,0])
sns.countplot(df["stalk-surface-below-ring"], ax=ax[6,1])
sns.countplot(df["stalk-color-above-ring"], ax=ax[7,0])
sns.countplot(df["stalk-color-below-ring"], ax=ax[7,1])
sns.countplot(df["veil-type"], ax=ax[8,0])
sns.countplot(df["veil-color"], ax=ax[8,1])
sns.countplot(df["ring-number"], ax=ax[9,0])
sns.countplot(df["ring-type"], ax=ax[9,1])
sns.countplot(df["spore-print-color"], ax=ax[10,0])
sns.countplot(df["population"], ax=ax[10,1])
sns.countplot(df["habitat"], ax[11,0])


for col in df.columns:
    print(df[col].value_counts())


# Looking at the column with missing data, what how many of the different
# values are poisonous or edible
sns.countplot(df["stalk-root"], hue=df["class"])

print(df["stalk-shape"].value_counts())

print(df["class"].value_counts())

sns.countplot(df["class"])




df3 = pd.DataFrame(dict(x=np.random.poisson(4, 500)))
df3.head()
ax = sns.barplot(x="x", y="x", data=df3, estimator=lambda x: len(x) / len(df) * 100)
ax.set(ylabel="Percent")


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

#dfz = df.drop(columns=["class"])
df.shape
df.head()
dfz = df
#dfz["class"] = (df["class"] == "e").astype(int)

dfz.head()


columns = dfz.columns[df.columns != "class"]

dfz = dfz[columns]

dfz = pd.get_dummies(dfz, columns=columns)

X = dfz
y = df["class"]

X.shape
y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log2 = LogisticRegression()
log2.fit(X_train, y_train)
y_pred = log2.predict(X_test)
print((y_pred == y_test).mean())
y_test.shape
y_pred.shape
log2.score(X_test, y_test)


print("Intercept: {:.2f}".format(log2.intercept_[0]))
print("Coefficients:")
name = 0
coefs = []
for i in log2.coef_[0]:
    print(" ",dfz.columns[name]+":", "{:.2f}".format(i))
    if(np.abs(i) > 2):
        coefs.append(name)
    name += 1

        
coefs = np.array(coefs)

coefs
dfz.shape
kk = dfz[dfz.columns[coefs]]
kk.shape

X_train, X_test, y_train, y_test = train_test_split(kk, y, test_size=0.3, random_state=42)

log3 = LogisticRegression()
log3.fit(X_train, y_train)
y_pred = log3.predict(X_test)
log3.score(X_test, y_test)

def getBestFeaturesLogCV(X_train, y_train):
    remaining = list(range(X_train.shape[1]))
    selected = []
    n = 1
    while len(selected) < n:
        # find the single features that works best in conjunction
        # with the already selected features
        accuracy_max = -1e7
        for i in remaining:
            selected.append(i)
            scores = cross_val_score(LogisticRegression(), X_train.iloc[:,selected], y_train, scoring='accuracy', cv=5)
            accuracy = scores.mean()
            selected.pop()
            if(accuracy > accuracy_max):
                accuracy_max = accuracy
                i_max = i
        remaining.remove(i_max)
        selected.append(i_max)
        print('num features: {}; accuracy: {:.2f};'.format(len(selected), accuracy_max))
    return selected

getBestFeaturesLogCV(X_train, y_train)


dfz.columns[27]



from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

df2 = df.drop(columns=["class"])
X = pd.get_dummies(df2, columns=df2.columns)
y = df["class"].values

X.shape
y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Baseline accuracy: {:.2f}".format(1-y_train.mean()))
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
clf.score(X_test, y_test)




