# This is the code for CodSoft Task 1: Movie Genre Classification
# The Code is written by Muhammad Mudassir Majeed
# The date is Dec-23
# Data Set: https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

#------------------------------------------------------------------------#


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Step 1: Read Data into Dataframes.
# The Data set is already split into train and test

data_train = pd.read_csv("train_data.txt", sep=":::", header= None, names= ["ID","Title","Genre","Description"] )
data_test = pd.read_csv("test_data_solution.txt",sep=":::", header= None, names= ["ID","Title","Genre","Description"])

# Step 2: Preleminary EDA
pd.set_option('display.max_columns', None)
data_train.head()
data_test.head()

# Step 2a: Check for null values
data_train.isnull().sum()
data_test.isnull().sum()
# No Null Values

# Step 2b: Look for class imbalances
data_train["Genre"].nunique() # 27 unique classes
data_test["Genre"].nunique()  # 27 unique classes

genre_train = data_train["Genre"].value_counts() 
print(genre_train)
genre_test = data_test["Genre"].value_counts()
print(genre_test)

# Severe class imbalance in both train and test data
plt.figure(figsize = (15,10))
sns.countplot(x=data_train["Genre"],order = genre_train.index,label="count")
plt.xticks(rotation = 45)
plt.show()

plt.figure(figsize = (15,10))
sns.countplot(x=data_test["Genre"],order = genre_test.index,label="count")
plt.xticks(rotation = 45)
plt.show()

# Note: Differrent pre-processing steps were tested, but the model 
        # Efficiency dropped. Things like text cleaning, stop words removal, 
        # punctuation removal, expanding contractions were all tested.
        # could not figure out a good pre-processing scheme
        # Running the code as is. 


# Step 3: Feature Extraction
# We will use TF-IDF technique

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(data_train['Description'])
X_test_tfidf = tfidf_vectorizer.transform(data_test['Description'])

# Define Target Variable
y_train = data_train['Genre']
y_test = data_test['Genre']

# Step 4: Model Training
# We will train a Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Predict Target values
y_predict = nb_classifier.predict(X_test_tfidf)


# Step 5: Model Evaluation
# We will measure Accuracy, Precision, Recall and F1 score.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Accuracy
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_test, y_predict, average='weighted')
print("Precision:", precision)

# Recall
recall = recall_score(y_test, y_predict, average='weighted')
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_test, y_predict, average='weighted')
print("F1 Score:", f1)

# Classification Report
class_report = classification_report(y_test, y_predict)
print("Classification Report:\n", class_report)
