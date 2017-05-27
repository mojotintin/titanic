import pandas as pd


def convert_age(age):
    return age // 5


def convert_sex(sex):
    if sex is 'male':
        return 0
    else:
        return 1


def convert_fare(fare):
    return fare // 10


def convert_embarked(embarked):
    if embarked == 'C':
        return 1
    elif embarked == 'Q':
        return 2
    elif embarked == 'S':
        return 3

train_frame = pd.read_csv('data/train.csv')
test_frame = pd.read_csv('data/test.csv')

columns_list = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
new_train_frame = train_frame.loc[:, columns_list]
columns_list.remove('Survived')
new_test_frame = test_frame.loc[:, columns_list]

new_train_frame['Age'] = new_train_frame['Age'].map(convert_age)
new_train_frame['Sex'] = new_train_frame['Sex'].map(convert_sex)
new_train_frame['Fare'] = new_train_frame['Fare'].map(convert_fare)
new_train_frame['Embarked'] = new_train_frame['Embarked'].map(convert_embarked)


new_test_frame['Age'] = new_test_frame['Age'].map(convert_age)
new_test_frame['Sex'] = new_test_frame['Sex'].map(convert_sex)
new_test_frame['Fare'] = new_test_frame['Fare'].map(convert_fare)
new_test_frame['Embarked'] = new_test_frame['Embarked'].map(convert_embarked)

new_train_frame.to_csv('data/new_train.csv')
new_test_frame.to_csv('data/new_test.csv')

print(new_test_frame.dtypes)
