import numpy as np
import pandas as pd
import sklearn.pipeline
import sklearn.feature_selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit


def newfeatures(data):
    data["Fam_size"] = data["SibSp"] + data["Parch"]
    data['Cabin'] = data['Cabin'].str[:1]
    return data

def title(x):
    x = x.split()
    r = 0
    for p,s in enumerate(x):
        if '.' in s:
            r = p        
    return x[r]

def agg_titles(x):
    title=x['Title']
    if title in ['Don.', 'Major.', 'Capt.', 'Jonkheer.', 'Rev.', 'Col.','Sir.']:
        return 'Mr.'
    elif title in ['Countess.', 'Mme.','L.','Dona.','Lady.']:
        return 'Mrs.'
    elif title in ['Mlle.', 'Ms.']:
        return 'Miss.'
    elif title =='Dr.':
        if x['Sex']=='Male':
            return 'Mr.'
        else:
            return 'Mrs.'
    else:
        return title


#Prepare training data
data = pd.read_csv("train.csv")
newfeatures(data)

data = data[data["Age"].isnull()==False]
data_labels = data["Survived"]

data['Title'] = data['Name'].apply(title)
data['Title'] = data.apply(agg_titles,axis=1)
avg_age = dict(round(data[data['Age'].isnull()==False].pivot_table(values='Age', columns='Title', aggfunc = 'mean')))
data.loc[data['Age'].isnull(),'Age'] = data['Title'].map(avg_age)

features_list = ['Pclass', 'Title', 'Sex','Age','Cabin','Fam_size','SibSp','Parch']
data = data[features_list]
data = pd.get_dummies(data, columns = ['Sex','Cabin','Title'])
del data['Cabin_T']

#Train
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()

steps = [('AdaBoost', clf)]
pipe = sklearn.pipeline.Pipeline(steps)

parameters = {'AdaBoost__n_estimators' : [40, 50, 70],
              'AdaBoost__learning_rate' : [0.3, 0.2, 0.1]}
                
sss = StratifiedShuffleSplit()
grid = GridSearchCV(pipe, parameters,cv = sss)
grid.fit(data, data_labels)
clf = grid.best_estimator_
print(clf)
print(grid.best_score_)

#Prepare test data
test = pd.read_csv("test.csv")
newfeatures(test)
test['Title'] = test['Name'].apply(title)
test['Title'] = test.apply(agg_titles,axis=1)
mean_age = round(data['Age'].mean())
test.loc[test['Age'].isnull(),'Age'] = test['Title'].map(avg_age)
test['Age'] = test['Age'].fillna(value= mean_age)

test = test[features_list]
test = pd.get_dummies(test, columns = ['Sex','Cabin','Title'])

#Predict and output in a format ready to submit
pred = pd.DataFrame(clf.predict(test),range(892,1310))
pred.index.name = "PassengerID"
pred.columns = ['Survived']
pred.to_csv('pred.csv')

