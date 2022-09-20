import pandas as pd
import numpy as np
import random as rnd

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

print ('Starting the execution ...')

pd.options.display.width=0
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

trainContest_df = pd.read_csv('input/train.csv')
testContest_df = pd.read_csv('input/test.csv')
test_df=trainContest_df[800:]
train_df= trainContest_df[:800]

combine = [train_df, test_df, testContest_df]


# data preprocessing
train_df['Ticket_Type']=train_df['Ticket'].str[0]
test_df['Ticket_Type']=test_df['Ticket'].str[0]
testContest_df['Ticket_Type']=testContest_df['Ticket'].str[0]
trainContest_df['Ticket_Type']=trainContest_df['Ticket'].str[0]

combine = [train_df, test_df, testContest_df, trainContest_df]

for dataset in combine: 
  dataset.loc[dataset['Cabin'].isna(), 'Cabin']=0

train_df['Cabin_Type']=train_df['Cabin'].str[0]
test_df['Cabin_Type']=test_df['Cabin'].str[0]
testContest_df['Cabin_Type']=testContest_df['Cabin'].str[0]
trainContest_df['Cabin_Type']=trainContest_df['Cabin'].str[0]

combine = [train_df, test_df, testContest_df, trainContest_df]

for dataset in combine: 
  dataset.loc[dataset['Cabin_Type'].isna(), 'Cabin_Type']=0

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
testContest_df= testContest_df.drop(['Ticket', 'Cabin'], axis=1)
trainContest_df= trainContest_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df, testContest_df, trainContest_df]

# extracting titles from the name field:
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# replace uncommon titles with common ones
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# converting the categorical titles to ordinal.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# dropping Name and PassengerId fields
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name', 'PassengerId'], axis=1)
testContest_PassengerId= testContest_df["PassengerId"]
testContest_df= testContest_df.drop(['Name', 'PassengerId'], axis=1)
trainContest_df= trainContest_df.drop(['Name', 'PassengerId'], axis=1)
combine = [train_df, test_df, testContest_df, trainContest_df]

# converting Sex feature to a new feature called Gender where female = 1 and male = 0
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

# filling in missing Age data
guess_ages = np.zeros((2, 3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

# Create Age bands which is going to be an ordinal field
#train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age']

# removing AgeBand field
#train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df, testContest_df, trainContest_df]

# create a new feature for FamilySize that combines Parch and SibSp
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# create IsAlone feature
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# dropping Parch and SibSp and FamilySize features
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
testContest_df= testContest_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
trainContest_df= trainContest_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df, testContest_df, trainContest_df]

# Creating artifical feature combining Pclass and Age
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

# fill missing values in Embarked with most common values
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# convert Embarked field to a new numeric port feature
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    dataset['Ticket_Type']= dataset['Ticket_Type'].map({'A':16, 'P':1, 'S':2, '1':3, '3':4, '2':5, 'C':6, '7':7, 'W':8, '4':9, 'F':10, 'L':11, '9':12, '6':13, '5':14, '8':15})

# filling in missing data in test dataset for Fare field
train_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
testContest_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
trainContest_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
# creating FareBand field
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

# convert Fare feature to ordinal values based on the FareBand
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

for dataset in combine: 
  dataset['ticketType*Fare']= dataset['Ticket_Type'] * dataset['Fare']

# convert Cabin_Type field to a new numeric port feature
for dataset in combine:
    dataset['Cabin_Type'] = dataset['Cabin_Type'].map({'D': 2, 'E': 2, 'B': 2, 'F': 2, 'C':2, 'G':1, 'A': 1})
    dataset.loc[(dataset['Cabin_Type'] !='2') & (dataset['Cabin_Type'] !='1') & (dataset['Cabin_Type']!=0), 'Cabin_Type'] = 0

train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df, testContest_df, trainContest_df]

X_train, X_test, Y_train, Y_test = train_test_split(train_df.drop("Survived", axis=1), train_df['Survived'], test_size=0.2, random_state=0)

# using LogisticRegression - score of 0.7775
logreg= LogisticRegression(max_iter=1000, random_state=42,)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_logreg = round(logreg.score(X_test, Y_pred) * 100, 2)
print("logreg Accuracy= " + str(logreg.score(X_test, Y_pred)))
print("Number of mislabeled points out of a total %d points : %d" % (Y_test.shape[0], (Y_test != Y_pred).sum()))

newX_test= test_df.drop("Survived", axis=1)
newY_test= test_df["Survived"]
newY_pred = logreg.predict(newX_test)
acc_logreg = round(logreg.score(newX_test, newY_pred) * 100, 2)
print("logreg Accuracy= " + str(logreg.score(newX_test, newY_pred)))
print("Number of mislabeled points out of a total %d points : %d" % (newY_test.shape[0], (newY_test != newY_pred).sum()))

# using Logreg on testContest data
ContestX_test= testContest_df

Y_pred = logreg.predict(ContestX_test)



# using RandomForest
random_forest= RandomForestClassifier(criterion='entropy', max_depth=10, n_estimators=300, random_state=34)
random_forest.fit(trainContest_df.drop('Survived', axis=1), trainContest_df['Survived'])
Y_pred2 = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_test, Y_pred2) * 100, 2)
print("RandomForest Accuracy= " + str(random_forest.score(X_test, Y_pred2)))
print("Number of mislabeled points out of a total %d points : %d" % (Y_test.shape[0], (Y_test != Y_pred2).sum()))

newX_test= test_df.drop("Survived", axis=1)
newY_test= test_df["Survived"]
newY_pred2 = random_forest.predict(newX_test)
acc_random_forest = round(random_forest.score(newX_test, newY_pred) * 100, 2)
print("random_forest Accuracy= " + str(random_forest.score(newX_test, newY_pred)))
print("Number of mislabeled points out of a total %d points : %d" % (newY_test.shape[0], (newY_test != newY_pred).sum()))

ContestX_test= testContest_df

Y_pred2 = random_forest.predict(ContestX_test)



# KNN
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, Y_train)
Y_pred3 = knn.predict(X_test)
acc_knn = round(knn.score(X_test, Y_pred3) * 100, 2)
print("KNN Accuracy= " + str(knn.score(X_test, Y_pred3)))
print("Number of mislabeled points out of a total %d points : %d" % (Y_test.shape[0], (Y_test != Y_pred3).sum()))

newX_test= test_df.drop("Survived", axis=1)
newY_test= test_df["Survived"]
newY_pred3 = knn.predict(newX_test)
acc_knn = round(knn.score(newX_test, newY_pred) * 100, 2)
print("random_forest Accuracy= " + str(knn.score(newX_test, newY_pred)))
print("Number of mislabeled points out of a total %d points : %d" % (newY_test.shape[0], (newY_test != newY_pred).sum()))

ContestX_test= testContest_df

Y_pred3 = knn.predict(ContestX_test)


result= pd.DataFrame()
result['PassengerId']=testContest_PassengerId
result["Survived"]=(Y_pred & Y_pred2)
#result['Survived']= result['Survived'].map({False:0, True:1})
result.to_csv("input/submit.csv", index=False)
