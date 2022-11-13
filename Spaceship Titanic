import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import norm
from matplotlib import pyplot as plt
import seaborn as sns 
train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')

combinedData= [train_df, test_df]

# changing homeplanet data: 0: Europa, 1: Earth, 2: Mars, 3: nan
for data in combinedData:
  data.loc[data['HomePlanet'].isna(), 'HomePlanet']=data['HomePlanet'].mode()[0]
  data.loc[data['HomePlanet']=='Europa', 'HomePlanet']=0
  data.loc[data['HomePlanet']=='Earth', 'HomePlanet']=1
  data.loc[data['HomePlanet']=='Mars', 'HomePlanet']=2
  data['HomePlanet']=data['HomePlanet'].astype(int)

# changing CryoSleep data: 0: false, 1: false, 2: nan
  data.loc[data['CryoSleep'].isna(), 'CryoSleep']=data['CryoSleep'].mode()[0]
  data.loc[data['CryoSleep']==0, 'CryoSleep']=0
  data.loc[data['CryoSleep']==1, 'CryoSleep']=1
  data['CryoSleep']=data['CryoSleep'].astype(int)

# changing VIP data: 0: false, 1: false, 2: nan
  data.loc[data['VIP'].isna(), 'VIP']=data['VIP'].mode()[0]
  data.loc[data['VIP']==0, 'VIP']=0
  data.loc[data['VIP']==1, 'VIP']=1
  data['VIP']=data['VIP'].astype(int)

# changing Age data: 0: 0-20, 1: 20-30, 3: 30-60, 4: >60
  data.loc[data['VIP']==0, 'VIP']=0
  data.loc[data['VIP']==1, 'VIP']=1
  data.loc[data['VIP'].isna(), 'VIP']=2
  data['VIP']=data['VIP'].astype(int)

# zero missing data in RoomService, FoodCourt, ShoppingMall, Spa and VRDeck. 
  data.loc[data['RoomService'].isna(), 'RoomService']=0
  data.loc[data['FoodCourt'].isna(), 'FoodCourt']=0
  data.loc[data['ShoppingMall'].isna(), 'ShoppingMall']=0
  data.loc[data['Spa'].isna(), 'Spa']=0
  data.loc[data['VRDeck'].isna(), 'VRDeck']=0

# Creating an aggregate field called Service1 that addes: RoomService, FoodCourt, ShoppingMall
  data['Service1']= data['RoomService'] + data['Spa'] + data['VRDeck']
  data['Service1']= data['Service1'].astype(int)

# Creating an aggregate field called Service12 that adds: Spa, VRDeck
  data['Service2']= data['FoodCourt'] + data['ShoppingMall']
  data['Service2']= data['Service2'].astype(int)

# Filling missing Age data with the majority which is 17
  data.loc[data['Age'].isna(), 'Age']=28
# make Age an ordinal field
  data['AgeGroup']= pd.cut(data['Age'],10, labels=[0,1,2,3,4,5,6,7,8,9])
  data['AgeGroup']=data['AgeGroup'].astype(int)

# Removing nan entries from Destination ['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e', 'Unknown']
  data.loc[data['Destination'].isna(), 'Destination']=data['Destination'].mode()[0]
  data.loc[data['Destination']=='TRAPPIST-1e', 'Destination']=0
  data.loc[data['Destination']=='PSO J318.5-22', 'Destination']=1
  data.loc[data['Destination']=='55 Cancri e', 'Destination']=2
  data['Destination']= data['Destination'].astype(int)

# Creating Group field from PassengerId
  data['GroupId']= data['PassengerId'].str.extract('([0-9][0-9][0-9][0-9])', expand=False)
  data['GroupId']= data['GroupId'].astype(int)
# Creating Deck and side fields from Cabin
  data['Deck']= data['Cabin'].str.extract('([A-Za-z])+', expand=False)
  data['Side']= data['Cabin'].str[-1]
  
  data.loc[data['Deck'].isna(), 'Deck']=data['Deck'].mode()[0]
  data.loc[data['Deck']=='B', 'Deck']=0
  data.loc[data['Deck']=='F', 'Deck']=1
  data.loc[data['Deck']=='A', 'Deck']=2
  data.loc[data['Deck']=='G', 'Deck']=3
  data.loc[data['Deck']=='E', 'Deck']=4
  data.loc[data['Deck']=='D', 'Deck']=5
  data.loc[data['Deck']=='C', 'Deck']=6
  data.loc[data['Deck']=='T', 'Deck']=7
  data['Deck']= data['Deck'].astype(int)

  data.loc[data['Side'].isna(), 'Side']=data['Side'].mode()[0]
  data.loc[data['Side']=='P', 'Side']=0
  data.loc[data['Side']=='S', 'Side']=1
  data['Side']= data['Side'].astype(int)

# Deleting original columns from TRAIN
train_df= train_df.drop(['Spa'], axis=1)
train_df= train_df.drop(['VRDeck'], axis=1)
train_df= train_df.drop(['RoomService'], axis=1)
train_df= train_df.drop(['FoodCourt'], axis=1)
train_df= train_df.drop(['ShoppingMall'], axis=1)
train_df= train_df.drop(['Age'], axis=1)
train_df= train_df.drop(['Name'], axis=1)
train_df= train_df.drop(['PassengerId'], axis=1)
train_df= train_df.drop(['Cabin'], axis=1)

# Deleting original columns TEST
test_df= test_df.drop(['Spa'], axis=1)
test_df= test_df.drop(['VRDeck'], axis=1)
test_df= test_df.drop(['RoomService'], axis=1)
test_df= test_df.drop(['FoodCourt'], axis=1)
test_df= test_df.drop(['ShoppingMall'], axis=1)
test_df= test_df.drop(['Age'], axis=1)
test_df= test_df.drop(['Name'], axis=1)
#test_df= test_df.drop(['PassengerId'], axis=1)
test_df= test_df.drop(['Cabin'], axis=1)

# changing Trasportation data: 0: false, 1: false on training
train_df.loc[train_df['Transported']==0, 'Transported']=0
train_df.loc[train_df['Transported']==1, 'Transported']=1
train_df['Transported']=train_df['Transported'].astype(int)

# Support Vector Machines
from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC
X_train, X_test, Y_train, Y_test = train_test_split(train_df.drop("Transported", axis=1), train_df['Transported'], test_size=0.2, random_state=0)

#svc = SVC(gamma='scale', random_state=100)
#svc.fit(X_train, Y_train)
#Y_pred = svc.predict(X_test)
#acc_svc = round(svc.score(X_test, Y_test) * 100, 2)
#print("SVM accuracy= " + str(acc_svc))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=65, random_state=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 2)
print("Random Forest Accuracy= " + str(acc_random_forest))

# Adaboost
from sklearn.ensemble import AdaBoostClassifier
adaboost= AdaBoostClassifier(n_estimators=87, random_state=100)
adaboost.fit(X_train, Y_train)
Y_pred= adaboost.predict(X_test)
adaboost.score(X_train, Y_train)
acc_adaboost = round(adaboost.score(X_test, Y_test) * 100, 2)
print("adaboost Accuracy= " + str(acc_adaboost))

result= pd.DataFrame()
Y_pred= pd.DataFrame()
Y_pred['p1']= adaboost.predict(test_df.drop('PassengerId', axis=1)).astype(int)
Y_pred['p2']= random_forest.predict(test_df.drop('PassengerId', axis=1)).astype(int)
#Y_pred['p3']= svc.predict(test_df.drop('PassengerId', axis=1)).astype(int)

Y_pred['p_aggreg']= Y_pred['p1'] + Y_pred['p2'] # + Y_pred['p3']

Y_pred.loc[Y_pred['p_aggreg']<1, 'p_final']=0
Y_pred.loc[Y_pred['p_aggreg']>=1, 'p_final']=1

Y_pred['p_final']= Y_pred['p_final'].astype(int)

Y_pred
result['PassengerId']=test_df['PassengerId']
result["Transported"]=Y_pred['p_final']
result['Transported']= result['Transported'].map({0:False, 1:True})
result.to_csv("input/submit.csv", index=False)
