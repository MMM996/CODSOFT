# This is the code for CodSoft Task 3: Customer Churn Prediction
# The Code is written by Muhammad Mudassir Majeed
# The date is Dec-23
# Data Set: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction

#----------------------------------------------------------------------------#

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


# Step 1: Load Data Set
dataset = pd.read_csv("data/Churn_Modelling.csv")
pd.set_option('display.max_columns', None)

# Step 2: Preliminary EDA
dataset.head()
dataset.info()

# Step 2a: Data Columns Check
dataset.columns

# Step 2b: Check for null values
dataset.isnull().sum() # No Null Values

# Step 2c: Check for duplicate values
duplicates = dataset.duplicated().sum()
print(duplicates)  # No Duplicate values

# Step 2d: Check for class imbalance
# Our target columns is Exited
dataset['Exited'].value_counts()
ax = sns.countplot(x=dataset["Exited"], label = "Count")


# Step 3: Data Pre-processing
# Step 3a: Drop Unnecessary columns
dataset.head()
# RowNumber, Surname and customer ID, I think all these are irrelevant
drop_col = ['RowNumber', 'CustomerId', 'Surname']
dataset = dataset.drop(drop_col, axis = 1)
dataset.head()
dataset.info()

# Step 3b: Encoding of categorical variables
# We have two object Data types, Gender and Geography
# First for Gender
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dataset['Gender'] = label_encoder.fit_transform(dataset['Gender'])
dataset.head(10)

# Second for Geography
# First check unique categories
dataset['Geography'].value_counts()  # Only three categories
# we can use both label encoding and one-hot encoding
dataset['Geography'] = label_encoder.fit_transform(dataset['Geography'])
dataset.head()
dataset.info()

# All values are now numeric


# Step 3c: Observe Each Attribute and identify any skewness
# we can plot histograms for each value and observe skewness

# We will create a grid
grid_rows, grid_col = 4,3

plt.figure(figsize = (10,15))

for i, column in enumerate(dataset.columns, 1):
    plt.subplot(grid_rows, grid_col, i)
    sns.histplot(data = dataset[column], kde= True)
    plt.title("histogram of {}".format(column))
    
plt.tight_layout()
plt.show()
# The only issue seems to be in Balance Column. We have a lot of zero values
plt.figure(figsize = (10,10))
sns.histplot(data= dataset['Balance'], kde = True)
# We can use Log transformation or Boxcox. But I think it is fine here

# Step 3d: Check and Remove Outliers

plt.figure(figsize=(10,15))

for i, column in enumerate(dataset.columns,1):
    plt.subplot(grid_rows, grid_col, i)
    sns.boxplot(data = dataset[column])
    plt.title("Box plot for {}".format(column))
    
plt.tight_layout()
plt.show()
# Age value has outliers. But this is normal. 

# Step 3e: Split data into input and target values
X = dataset.drop(['Exited'], axis =1 )
Y = dataset['Exited']

# Step 3f: Standardization or Normalization of data
X.describe()
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_standardized = scalar.fit_transform(X)
X_stand_df = pd.DataFrame(data =X_standardized, columns= X.columns)
X_stand_df.describe()


# Step 3g: Resolve class imbalances
# we can use SMOTE technique
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state =42)
Y.value_counts()

X_resampled, Y_resampled = smote.fit_resample(X_standardized, Y)
Y_resampled.value_counts()

X_resample_df = pd.DataFrame(data = X_resampled, columns = X_stand_df.columns)
Y_resample_df = pd.DataFrame(data = Y_resampled)

# Step 4: Post EDA
# Step 4a: Observe violin plots

data_2 = pd.concat([Y_resample_df,X_resample_df], axis = 1)
data_3 = pd.melt(data_2, id_vars='Exited', var_name = 'features', value_name= 'value')
plt.figure(figsize=(20,20))
sns.violinplot(data = data_3, x= 'features', y='value', split = True, 
               inner = 'quart', hue = 'Exited')
# Gender, is_Active Member, has_Card all does not seem very important


# Step 4b: Observe Correlation heat map
sns.set(style = 'whitegrid', palette= 'muted')
f, ax = plt.subplots(figsize =(10,10))
sns.heatmap(X_resample_df.corr(), annot = True, linewidths=.5,fmt='.1f', ax = ax)
# Very low correlation values

# This cocludes our Pre-processing

# Step 5: Feature Extraction 
# All features are good


# Step 6: Model Training and Evaluation
# Step 6a: Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_resample_df, Y_resample_df,
                                                    test_size= 0.2,random_state=42)

# Step 6b: Train Models
# Step 6b1: Logisitc Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_logreg = logreg.predict(X_test)

# Step 6b2: Random Forest
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=100, random_state=42)
RF.fit(X_train, Y_train)
Y_pred_RF = RF.predict(X_test)

# Step 6b3: Gradeint Boosting
from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier(n_estimators=100, random_state=42)
GB.fit(X_train, Y_train)
Y_pred_GB = GB.predict(X_test)

# Step 6c: Evaluate and Compare Models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Accuracy
accuracy_logreg = accuracy_score(Y_test, Y_pred_logreg)
accuracy_GB = accuracy_score(Y_test, Y_pred_GB)
accuracy_RF = accuracy_score(Y_test, Y_pred_RF)

print("Accuracy for Logistic regression: {:.2f}".format(accuracy_logreg))
print("Accuracy for Gradient Boosting: {:.2f}".format(accuracy_GB))
print("Accuracy for Random Forest: {:.2f}\n".format(accuracy_RF))

# Precision
precision_logreg = precision_score(Y_test, Y_pred_logreg)
precision_GB = precision_score(Y_test, Y_pred_GB)
precision_RF = precision_score(Y_test, Y_pred_RF)

print("Precision for Logistic regression: {:.2f}".format(precision_logreg))
print("Precision for Gradient Boosting: {:.2f}".format(precision_GB))
print("Precision for Random Forest: {:.2f}\n".format(precision_RF))
# Recall
recall_logreg = recall_score(Y_test, Y_pred_logreg)
recall_GB = recall_score(Y_test, Y_pred_GB)
recall_RF = recall_score(Y_test, Y_pred_RF)

print("Recall for Logistic regression: {:.2f}".format(recall_logreg))
print("Recall for Gradient Boosting: {:.2f}".format(recall_GB))
print("Recall for Random Forest: {:.2f}\n".format(recall_RF))

# F1 Score
f1_logreg = f1_score(Y_test, Y_pred_logreg)
f1_GB = f1_score(Y_test, Y_pred_GB)
f1_RF = f1_score(Y_test, Y_pred_RF)

print("f1 score for Logistic regression: {:.2f}".format(f1_logreg))
print("f1 score for Gradient Boosting: {:.2f}".format(f1_GB))
print("f1 score for Random Forest: {:.2f}\n".format(f1_RF))

# Display Data in a Table
from tabulate import tabulate
# Output data
data = [
    ['Logistic regression', accuracy_logreg, precision_logreg, recall_logreg, f1_logreg],
    ['Gradient Boosting', accuracy_GB, precision_GB, recall_GB, f1_GB],
    ['Random Forest', accuracy_RF, precision_RF, recall_RF, f1_RF]
]

# Column headers
headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']

# Create and print the table
table = tabulate(data, headers, tablefmt='fancy_grid')
print(table)