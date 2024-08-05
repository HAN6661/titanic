# titanic
## machine learning to create a model that predicts which passengers survived the Titanic or not Using Logistic Regression to train the model, and accuracy score to evaluate the model
### We import the libraries  

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

### Reading the data
```
data = pd.read_csv('/content/train.csv')
```

### pd.read_csv to load the data from csv file to Pandas DataFrame:

### this will print first 5 rows in the dataset 
```
data.head()
```

![Screenshot 2024-08-05 092301](https://github.com/user-attachments/assets/9dcb3f26-7244-4042-a759-a044502ef977)


### number of rows and columns
```
data.shape
```

### the info command to learn more about the data, such as the number of rows and columns, data types, and the number of missing values.
```
data.info()
```

![Screenshot 2024-08-05 092545](https://github.com/user-attachments/assets/f3879680-f98c-459a-8f7c-316c7de25b35)

### to view the Missing valuse in each column:
```
data.isnull().sum()
```

![Screenshot 2024-08-05 092640](https://github.com/user-attachments/assets/1701dce3-5077-42f8-9d03-e5600eac7467)


### simply use 'filllna' function, or any other way such as SimpleImputer
```
data['Age'].fillna(data['Age'].mean(), inplace=True)
```

### number of missing values
```
data['Age'].isnull().sum()
```

### drop this column from the dataset
```
data = data.drop(['Cabin'], axis=1)
```

```
data.head()
```

![Screenshot 2024-08-05 092717](https://github.com/user-attachments/assets/2b55266f-6f61-458a-93ce-7a54fcbac151)

### Embarked column, there are only two missing values
```
data['Embarked'].value_counts()
```

### fill the missing values in Embarked with the mode of Embarked column:
```
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
```

```
data['Embarked'].isnull().sum()
```

### Drop the PassengerId and Name Columns from the dataset:
```
data = data.drop(['PassengerId', 'Name'], axis=1)
```

```
data.head()
```

![Screenshot 2024-08-05 092757](https://github.com/user-attachments/assets/ee65b1a4-2ef3-45f1-9b64-9ab5befe9a63)


### In Age column we will replace all male values with 0 and all the female values with 1.
and we will do the same in Embarked column: S=> 0 , C=> 1, Q => 2

```
data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

data.head()
```

![Screenshot 2024-08-05 092818](https://github.com/user-attachments/assets/24d1579c-27e4-4bf6-beab-e1e26bc8d2f5)


### check if there are duplicates in the dataset:
```
num_duplicates = data.duplicated().sum()
```

### drop the duplicates:
```
data = data.drop_duplicates()
data.describe()
```

![Screenshot 2024-08-05 092853](https://github.com/user-attachments/assets/ce8b47a4-e111-4568-b8de-ff49103c64c8)

### Check for non-numeric columns
```
non_numeric_cols = data.select_dtypes(include=['object']).columns

print(non_numeric_cols)
```
### Display unique values in non-numeric columns
```
for col in non_numeric_cols:
    print(f"Unique values in {col}:")
    print(data[col].unique())

```

### Attempt to convert columns to numeric, if applicable
```
for col in non_numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')  # Coerce errors to NaN
```

### Drop rows with non-numeric values
```
data = data.dropna()
```

### Check data types again
```
print(data.dtypes)
```
### Recalculate correlations
```
print(data.corr()['Survived'])
```

### to understand more about data lets find the number of people survived and not survived
```
data['Survived'].value_counts()
```

### making a count plot for 'Survived' column
```
sns.countplot(x='Survived', data=data)
```

### making a count plot for 'Sex' column
```
sns.countplot(x='Sex', data=data)
```

### now lets compare the number of survived beasd on the gender
```
sns.countplot(x='Sex', hue='Survived', data=data)
```

### now lets compare the number of survived beasd on the Pclass
```
sns.countplot(x='Pclass', hue='Survived', data=data)
```


### Separating features and target so that we can prepare the data for training machine learning models. In the Titanic dataset, the Survived column is the target variable, and the other columns are the features.
```
x = data.drop(columns = ['Survived'], axis=1)
y = data['Survived']
```

### Split the data into training data & Testing data using train_test_split function :
```
from sklearn.model_selection import train_test_split
```

### Separate features and target
```
X = data.drop('Survived', axis=1)
y = data['Survived']
```

### Split the data
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Output the shapes
```
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```































