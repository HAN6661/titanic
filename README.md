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
#### pd.read_csv to load the data from csv file to Pandas DataFrame:

### this will print first 5 rows in the dataset 
```
data.head()
```
### number of rows and columns
```
data.shape
```

### the info command to learn more about the data, such as the number of rows and columns, data types, and the number of missing values.
```
data.info()
```

### to view the Missing valuse in each column:
```
data.isnull().sum()
```

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

### Embarked column, there are only two missing values
```
data['Embarked'].value_counts()
```

### 






































