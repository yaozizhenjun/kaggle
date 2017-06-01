

import pandas as pd 
import numpy as np 

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#load the data

titanic_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

titanic_df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
test_df.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
#Embarked
titanic_df.Embarked = titanic_df.Embarked.fillna('C')
embarked_dummies_titanic = pd.get_dummies(titanic_df.Embarked)
titanic_df = titanic_df.join(embarked_dummies_titanic)
embarked_dummies_test = pd.get_dummies(test_df.Embarked)
test_df = test_df.join(embarked_dummies_test)

titanic_df.drop('Embarked',axis=1,inplace=True)
test_df.drop('Embarked',axis=1,inplace=True)

#SibSp and Parch   => Family
titanic_df['Family'] = titanic_df.SibSp + titanic_df.Parch
test_df['Family'] = test_df.SibSp + test_df.Parch

titanic_df.drop(['SibSp','Parch'],axis=1,inplace=True)
test_df.drop(['SibSp','Parch'],axis=1,inplace=True)

#Age 
average_age_titanic = titanic_df.Age.mean()
std_age_titanic = titanic_df.Age.std()
count_nan_age_titanic = titanic_df.Age.isnull().sum()

average_age_test = test_df.Age.mean()
std_age_test = test_df.Age.std()
count_nan_age_test = test_df.Age.isnull().sum()

rand_1 = np.random.randint(average_age_titanic - std_age_titanic,average_age_titanic + std_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test,average_age_test + std_age_test)

titanic_df.Age[np.isnan(titanic_df.Age)] = rand_1
test_df.Age[np.isnan(test_df.Age)] = rand_2

titanic_df.Age = titanic_df.Age.astype(int)
test_df.Age = test_df.Age.astype(int)

#Fare
test_df.Fare.fillna(test_df.Fare.median(),inplace=True)
titanic_df.Fare = titanic_df.Fare.astype(int)
test_df.fare = test_df.Fare.astype(int)

#Age and Sex => Person
def get_person(passenger):
	age,sex  = passenger
	return 'child' if age < 16 else sex

titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person,axis=1)
test_df['Person'] = test_df[['Age','Sex']].apply(get_person,axis=1)

titanic_df.drop('Sex',axis=1,inplace=True)
test_df.drop('Sex',axis=1,inplace=True)

person_dummies_titanic = pd.get_dummies(titanic_df.Person)
person_dummies_titanic.columns = ['Child','Male','Female']
titanic_df = titanic_df.join(person_dummies_titanic)
titanic_df.drop('Person',axis=1,inplace=True)

person_dummies_test = pd.get_dummies(test_df.Person)
person_dummies_test.columns = ['Child','Male','Female']
test_df = test_df.join(person_dummies_test)
test_df.drop('Person',axis=1,inplace=True)

#define training and testing sets
X_train = titanic_df.drop('Survived',axis=1)
Y_train = titanic_df.Survived
X_test = test_df.drop('PassengerId',axis=1).copy()

#RandomForestClassifier
'''random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,Y_train)
Y_pred = random_forest.predict(X_test)
print random_forest.score(X_train,Y_train)
'''

#SVM
svc = SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print svc.score(X_train,Y_train)

submission = pd.DataFrame({'PassengerId' : test_df.PassengerId , 'Survived':Y_pred})
submission.to_csv('my_titanic.csv',index=False)
#print titanic_df.info()