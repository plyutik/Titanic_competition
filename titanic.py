# Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
import warnings
warnings.filterwarnings('ignore')

# Load the train and test datasets to create two DataFrames
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Print the `head` of the train and test dataframes
print(train.head())
print(test.head())

# Passengers that survived vs passengers that passed away
print(train["Survived"].value_counts())
print(train["Survived"].value_counts(normalize = True))

# Males that survived vs males that passed away
print(train["Survived"][train["Sex"] == 'male'].value_counts())
print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True))

# Females that survived vs Females that passed away
print(train["Survived"][train["Sex"] == 'female'].value_counts())
print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True))

# Chart of survival vs passenger sex
survived_class = pd.crosstab(index=train["Survived"],
                            columns=train["Sex"])   # Include row and column totals
survived_class.columns = ["Female","Male"]
survived_class.index= ["Died","Survived"]
survived_class.plot(kind="bar",
                 figsize=(8,8),
                 stacked=True)
plt.show()

# Correlate the survival with the age variable.
train['Age'].fillna((train['Age'].mean()), inplace=True)
figure = plt.figure(figsize=(15,6))
plt.hist([train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
plt.show()

# Focus on the Fare ticket of each passenger and correlate it with the survival
figure = plt.figure(figsize=(15,6))
plt.hist([train[train['Survived']==1]['Fare'],train[train['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
plt.show()

# How the embarkation site affects the survival
embark = train[train['Survived'] == 1]['Embarked'].value_counts()
dead_embark = train[train['Survived'] == 0]['Embarked'].value_counts()
df = pd.DataFrame([embark,dead_embark])
df.index = ['Survived','Dead']
df.plot(kind='bar', stacked=True, figsize=(15,6))
plt.show()

# Convert the male and female groups to integer form into train
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

# Convert the male and female groups to integer form into test
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

# Impute the Embarked variable into train
train["Embarked"] = train["Embarked"].fillna("S")

# Convert the Embarked classes to integer form into train
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# Impute the Embarked variable into test
test["Embarked"] = test["Embarked"].fillna("S")

# Convert the Embarked classes to integer form into test
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

# Print the train data to see the available features and clean data
train = train.drop(['Cabin'], axis=1)
train['Age'].fillna((train['Age'].mean()), inplace=True)
print(train)

# Print the train data to see the available features and clean data
test = test.drop(['Cabin'], axis=1)
test['Age'].fillna((test['Age'].mean()), inplace=True)
print(test)

# Building  my_forest
y_train = train['Survived']
X_train = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
test_features = Imputer().fit_transform(test_features)

# Fit model and print score
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(X_train, y_train)
print(my_forest.score(X_train, y_train))

# Make prediction
prediction = my_forest.predict(test_features)

# Write a solution on csv file
solution = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })
solution.to_csv('solution.csv', index=False)