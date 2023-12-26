# This is the code for CodSoft Task 2: Credit card fraud Detection
# The Code is written by Muhammad Mudassir Majeed
# The date is Dec-23
# Data Set: https://www.kaggle.com/datasets/kartik2112/fraud-detection/data
# Speccial Thanks to:
    # https://www.kaggle.com/code/islamashraaf/credit-card-fraud-detection#Model-Building
    # The code is well written and teaches a lot of new techniques

#----------------------------------------------------------------------------#

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load Data Set
data_train = pd.read_csv("Data/fraudTrain.csv")
data_test = pd.read_csv("Data/fraudTest.csv")

# Step 2: Preliminary EDA
pd.set_option('display.max_columns', None)
data_train.head(20)
data_train.info()
fraudulent_rows = data_train[data_train['is_fraud'] == 1]
fraudulent_rows.head(20)
# We can see from fradulent rows that fraudulent names, job all repeat.

# Step 2a: Data Columns Check
data_train.columns

# Step 2b: Check for null values
data_train.isnull().sum()
data_test.isnull().sum()
# No Null values

# Step 2c: Check for class imbalances
plt.figure(figsize=(15,10))
sns.countplot(x =data_train["is_fraud"], label = 'Count')
sns.countplot(x= data_test["is_fraud"], label = 'Count')
# Severe class imbalance

# Step 3: Data Pre-processing
# Step 3a: Make new features and drop redundant or unwanted columns
# I am asuming that fradulent transactions are:
                # Carried out under similar or same names
                # Age is important
                # Shopping category is important
                # Amount is important
                # Location is important
                # Job is important
                # Merchant info is important
                
        # There may be other factors like fraud transacations could be happening at a particular time
        # Or at a particular day of the week etc. Currently ignoring these factors
        
        # We can preprocess data to remove other unwanted columns
        # Similarly we can use exsiting features to make new useful features
                # We can make Age
                # We can combine first nad last name etc.
                

# Location values are very redundant
# Like city population, lat, long, City, State. All these effects
# Can be looked up from zip code.
drop_col = ['Unnamed: 0','lat','long','city_pop','city','state', 'street']

# Convert date of birth to age.
data_train["trans_date_trans_time"] = pd.to_datetime(data_train["trans_date_trans_time"])
data_train['dob'] = pd.to_datetime(data_train['dob'])
data_train['age'] = np.ceil((data_train['trans_date_trans_time'] - data_train['dob']).dt.days / 365).astype(int)

data_test["trans_date_trans_time"] = pd.to_datetime(data_test["trans_date_trans_time"])
data_test['dob'] = pd.to_datetime(data_test['dob'])
data_test['age'] = np.ceil((data_test['trans_date_trans_time'] - data_test['dob']).dt.days / 365).astype(int)
drop_col.extend(["dob","trans_date_trans_time"])

# Combine first and last names
data_train['name']= data_train['first'] + " " + data_train['last']
data_test['name']= data_test['first'] + " " + data_test['last']
drop_col.extend(["first","last"])

# I am also dropping transaction number
drop_col.extend(["trans_num"])

# Now drop all redundant and unwanted columns
data_train.drop(drop_col, inplace = True, axis = 1)
data_test.drop(drop_col, inplace = True, axis = 1)
data_train.head()
data_test.head()


# Step 3b: Encoding of categorical variables
# We want to encode category, merchant, Gender, name and job columns
# Count unique values for each column to get an idea
unique_train = data_train.apply(lambda col: col.nunique())
unique_test = data_test.apply(lambda col: col.nunique())
print(unique_train)
print(unique_test)
data_train.info()
data_test.info()

# We can easily encode our categorical data using One hot Encoding
    # However it will take a lot of memory
# Label Encoder is a good option but it works for nominal data
from category_encoders import WOEEncoder

# Initialize WOEEndoer
WOEencoder = WOEEncoder()

# Encode categorical columns
columns_to_encode = ['category', 'merchant', 'gender', 'name', 'job',]

for col in columns_to_encode:
    data_train[col] = WOEencoder.fit_transform(data_train[col],data_train['is_fraud'])
    data_test[col] = WOEencoder.fit_transform(data_test[col],data_test['is_fraud'] )

# Display the resulting DataFrame
data_train.head()
data_test.head()

# We can observe that ampount data is highly skewed. We need to address this
# Step 3c: Handling Data Skewness
# We can apply log transformations
data_train['amt'] = np.log1p(data_train['amt']) 
data_test['amt'] = np.log1p(data_test['amt']) 


# Step 3d: Split into input and Target Sets
X_train = data_train.drop(['is_fraud'], axis =1)
X_test = data_test.drop(['is_fraud'], axis =1)
Y_train = pd.DataFrame(data_train["is_fraud"])
Y_test = pd.DataFrame(data_test["is_fraud"])
X_train.describe()
X_test.describe()


# Step 3e: Standardization or Normalization of data
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
X_train_str = scalar.fit_transform(X_train)
X_train_df = pd.DataFrame(X_train_str, columns= X_train.columns)
X_test_str = scalar.fit_transform(X_test)
X_test_df = pd.DataFrame(X_test_str,columns=X_test.columns)

# Step 3f: Check and Resolve class imbalances
# WE will use smote methodolgy
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
Y_train.value_counts()

X_train_resampled, Y_train_Resampled = smote.fit_resample(X_train_df, Y_train)
X_resampled_df = pd.DataFrame(X_train_resampled)
Y_resampled_df = pd.DataFrame(Y_train_Resampled)
Y_resampled_df.value_counts()

# Step 4: Post EDA
# Step 4a: Observe violin plots
data_2 = pd.concat([Y_resampled_df, X_resampled_df], axis = 1)
data_3 = pd.melt(data_2 , id_vars = 'is_fraud', var_name='features', value_name='value')
sns.violinplot(data= data_3 , x= 'features', y='value', hue='is_fraud',
               inner = 'quart', split = True)
plt.xticks(rotation = 45)
plt.show()

# Step 4c: Observe Correlation heat map
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(X_resampled_df.corr(), annot = True, linewidths=0.5, fmt = '.1f', ax=ax)

# Step 6: Model Training and Evaluation
X_train = X_resampled_df
X_test = X_test_df
Y_train = Y_resampled_df
Y_test = Y_test


# Step 6b1: Logisitc Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_logreg = logreg.predict(X_test)

# Step 6b2: Random Forest
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=10, random_state=42)
RF.fit(X_train, Y_train)
Y_pred_RF = RF.predict(X_test)

# Step 6b3: Gradeint Boosting
from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier(n_estimators=10, random_state=42)
GB.fit(X_train, Y_train)
Y_pred_GB = GB.predict(X_test)

# Step 6b4: Decision Trees
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(X_train, Y_train)
Y_pred_DT = DT.predict(X_test)

# Step 6c: Evaluate and Compare Models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Accuracy
accuracy_logreg = accuracy_score(Y_test, Y_pred_logreg)
accuracy_GB = accuracy_score(Y_test, Y_pred_GB)
accuracy_RF = accuracy_score(Y_test, Y_pred_RF)
accuracy_DT = accuracy_score(Y_test, Y_pred_DT)

print("Accuracy for Logistic regression: {:.2f}".format(accuracy_logreg))
print("Accuracy for Gradient Boosting: {:.2f}".format(accuracy_GB))
print("Accuracy for Random Forest: {:.2f}".format(accuracy_RF))
print("Accuracy for Decision Tree: {:.2f}\n".format(accuracy_DT))

# Precision
precision_logreg = precision_score(Y_test, Y_pred_logreg)
precision_GB = precision_score(Y_test, Y_pred_GB)
precision_RF = precision_score(Y_test, Y_pred_RF)
precision_DT = precision_score(Y_test, Y_pred_DT)

print("Precision for Logistic regression: {:.2f}".format(precision_logreg))
print("Precision for Gradient Boosting: {:.2f}".format(precision_GB))
print("Precision for Random Forest: {:.2f}".format(precision_RF))
print("Precision for Decision Tree: {:.2f}\n".format(precision_DT))

# Recall
recall_logreg = recall_score(Y_test, Y_pred_logreg)
recall_GB = recall_score(Y_test, Y_pred_GB)
recall_RF = recall_score(Y_test, Y_pred_RF)
recall_DT = recall_score(Y_test, Y_pred_DT)

print("Recall for Logistic regression: {:.2f}".format(recall_logreg))
print("Recall for Gradient Boosting: {:.2f}".format(recall_GB))
print("Recall for Random Forest: {:.2f}".format(recall_RF))
print("Recall for Decision Tree: {:.2f}\n".format(recall_DT))

# F1 Score
f1_logreg = f1_score(Y_test, Y_pred_logreg)
f1_GB = f1_score(Y_test, Y_pred_GB)
f1_RF = f1_score(Y_test, Y_pred_RF)
f1_DT = f1_score(Y_test, Y_pred_DT)

print("f1 score for Logistic regression: {:.2f}".format(f1_logreg))
print("f1 score for Gradient Boosting: {:.2f}".format(f1_GB))
print("f1 score for Random Forest: {:.2f}".format(f1_RF))
print("f1 score for Decision Tree: {:.2f}\n".format(f1_DT))

# Display Data in a Table
from tabulate import tabulate
# Output data
data = [
    ['Logistic regression', accuracy_logreg, precision_logreg, recall_logreg, f1_logreg],
    ['Gradient Boosting', accuracy_GB, precision_GB, recall_GB, f1_GB],
    ['Random Forest', accuracy_RF, precision_RF, recall_RF, f1_RF],
    ['Decision Tree', accuracy_DT, precision_DT, recall_DT, f1_DT]
]

# Column headers
headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']

# Create and print the table
table = tabulate(data, headers, tablefmt='fancy_grid')
print(table)

# Step 7: Optimize and tune best model



