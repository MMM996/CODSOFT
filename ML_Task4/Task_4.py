# This is the code for CodSoft Task 4: Spam SMS Detection
# The Code is written by Muhammad Mudassir Majeed
# The date is Dec-23
# Data Set: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

#------------------------------------------------------------------------#

import pandas as pd
import seaborn as sns
import numpy as np


# Step 1: Load Data set
# we are recieving a uyt-8 codec error. We will try to detect coding of this file.
import chardet
with open('data/spam.csv', 'rb') as file:
    result = chardet.detect(file.read())

encoding = result['encoding']

dataset= pd.read_csv("data/spam.csv",encoding= encoding)

# Step 2: Preleminary EDA
pd.set_option('display.max_columns', None)
dataset.head()
drop_col = ["Unnamed: 2","Unnamed: 3", "Unnamed: 4"]
dataset = dataset.drop(drop_col, axis = 1)
dataset.head()
# We will rename columns
dataset.rename(columns ={"v1": "class", "v2": "message"}, inplace = True)

# Step 2a: Check for null values
dataset.isnull().sum()

# Step 2b: Check for Duplicate values
dataset.duplicated().sum()  # 403 Duplicate rows

# Step 2c: Look for class imbalances
dataset['class'].value_counts()
sns.countplot(x=dataset['class'], label= 'count')
# We have class imbalance. 
  

# Step 3: Pre-Processing NLP
# Step 3a: Open Contractions
import contractions
# This is a library that provides a list of all the contractions

dataset['message_expand'] = dataset['message'].apply(lambda x: [contractions.fix(word) for word in x.split()])

# We will not perform a lot of data pre-processing
        # Removing punctuation: Spam sms may have lot of those like exclamations
        # Similarly I am not removing stop words
        # We are not performing any text cleaning like html remove, etc.

# Step 3b: Convert all characters to lower case.
dataset['message_expand'] = dataset['message_expand'].apply(lambda tokens: [word.lower() for word in tokens])

# Step 3c: Split in target and input features
# Input Attributes
X = dataset['message_expand']
# Target Attributes
Y = dataset['class']

# Step 3g: Resolve class imbalances
# Skipping for now. We will use stratified sampling to handle this.
# We will also use ensemble methods for prediction to combat this issue.

# Step 4: Feature Extraction
# Step 4a: Make TF-IDF Vectors
# First combine the tokens to make sentences again
X = X.str.join(" ")

# Apply TF-IDF Vecorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=1)
X_tfidf = tfidf.fit_transform(X,Y)

# Step 4b: Use word2vec approach
from gensim.models import Word2Vec
# Word2Vec Model
word2vec_model = Word2Vec(sentences=X, vector_size=100,
                          window=5, min_count=1, workers=4)

# Word2Vec Vectorization
def word2vec_transform(text, model):
    vectors = []
    for word in text:
        if word in model.wv:
            vectors.append(model.wv[word])
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

X_word2vec = np.array([word2vec_transform(tokens, word2vec_model) 
                             for tokens in X])

# This will help us compare results for both techniques

# Step 5: Model Training and Evaluation
# Step 5a: Split model into train and test sets
from sklearn.model_selection import train_test_split
X_train_tfidf, X_test_tfidf, Y_train_tfidf, Y_test_tfidf = train_test_split(X_tfidf,Y,
                                test_size=0.2,random_state=42, stratify=Y)

X_train_word, X_test_word, Y_train_word, Y_test_word = train_test_split(X_word2vec,Y,
                                test_size=0.2,random_state=42, stratify=Y)

# Step 5b: Train Models 
# Step 5b1: Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg_tfidf = LogisticRegression()
logreg_tfidf.fit(X_train_tfidf, Y_train_tfidf)
Y_predict_tfidf_logreg = logreg_tfidf.predict(X_test_tfidf)

logreg_word = LogisticRegression()
logreg_word.fit(X_train_word, Y_train_word)
Y_predict_word_logreg = logreg_word.predict(X_test_word)

# Step 5b2: Naive Bayes
from sklearn.naive_bayes import MultinomialNB

NB_tfidf = MultinomialNB()
NB_tfidf.fit(X_train_tfidf, Y_train_tfidf)
Y_predict_tfidf_NB = NB_tfidf.predict(X_test_tfidf)

# Naive Bayes or MultinomialNB requires only positive values
# word2vec has both postive and negative, So, cant be used.


# Step 5b3: SVM
from sklearn.svm import SVC

SVM_tfidf = SVC(kernel='linear', C=1.0)
SVM_tfidf.fit(X_train_tfidf, Y_train_tfidf)
Y_predict_tfidf_SVM = SVM_tfidf.predict(X_test_tfidf)

SVM_word = SVC(kernel='linear', C=1.0)
SVM_word.fit(X_train_word, Y_train_word)
Y_predict_word_SVM = SVM_word.predict(X_test_word)

# Step 5b4: Random Forest
from sklearn.ensemble import RandomForestClassifier
RF_tfidf = RandomForestClassifier(n_estimators= 100, random_state= 42)
RF_tfidf.fit(X_train_tfidf, Y_train_tfidf)
Y_predict_tfidf_RF = RF_tfidf.predict(X_test_tfidf)

RF_word = RandomForestClassifier(n_estimators= 100, random_state= 42)
RF_word.fit(X_train_word, Y_train_word)
Y_predict_word_RF = RF_word.predict(X_test_word)

# Step 5c: Compare models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Accuracy
accuracy_logreg_tfidf = accuracy_score(Y_test_tfidf, Y_predict_tfidf_logreg)
accuracy_NB_tfidf = accuracy_score(Y_test_tfidf, Y_predict_tfidf_NB)
accuracy_SVM_tfidf = accuracy_score(Y_test_tfidf, Y_predict_tfidf_SVM)
accuracy_RF_tfidf = accuracy_score(Y_test_tfidf, Y_predict_tfidf_RF)
accuracy_logreg_word = accuracy_score(Y_test_tfidf, Y_predict_word_logreg)
accuracy_SVM_word = accuracy_score(Y_test_tfidf, Y_predict_word_SVM)
accuracy_RF_word = accuracy_score(Y_test_tfidf, Y_predict_word_RF)


# Precision
precision_logreg_tfidf = precision_score(Y_test_tfidf, Y_predict_tfidf_logreg, pos_label = 'spam')
precision_NB_tfidf = precision_score(Y_test_tfidf, Y_predict_tfidf_NB, pos_label = 'spam')
precision_SVM_tfidf = precision_score(Y_test_tfidf, Y_predict_tfidf_SVM, pos_label = 'spam')
precision_RF_tfidf = precision_score(Y_test_tfidf, Y_predict_tfidf_RF, pos_label = 'spam')
precision_logreg_word = precision_score(Y_test_tfidf, Y_predict_word_logreg, pos_label = 'spam')
precision_SVM_word = precision_score(Y_test_tfidf, Y_predict_word_SVM, pos_label = 'spam')
precision_RF_word = precision_score(Y_test_tfidf, Y_predict_word_RF, pos_label = 'spam')

# Recall
recall_logreg_tfidf = recall_score(Y_test_tfidf, Y_predict_tfidf_logreg, pos_label = 'spam')
recall_NB_tfidf = recall_score(Y_test_tfidf, Y_predict_tfidf_NB, pos_label = 'spam')
recall_SVM_tfidf = recall_score(Y_test_tfidf, Y_predict_tfidf_SVM, pos_label = 'spam')
recall_RF_tfidf = recall_score(Y_test_tfidf, Y_predict_tfidf_RF, pos_label = 'spam')
recall_logreg_word = recall_score(Y_test_tfidf, Y_predict_word_logreg, pos_label = 'spam')
recall_SVM_word = recall_score(Y_test_tfidf, Y_predict_word_SVM, pos_label = 'spam')
recall_RF_word = recall_score(Y_test_tfidf, Y_predict_word_RF, pos_label = 'spam')

# F1 Score
f1_logreg_tfidf = f1_score(Y_test_tfidf, Y_predict_tfidf_logreg, pos_label = 'spam')
f1_NB_tfidf = f1_score(Y_test_tfidf, Y_predict_tfidf_NB, pos_label = 'spam')
f1_SVM_tfidf = f1_score(Y_test_tfidf, Y_predict_tfidf_SVM, pos_label = 'spam')
f1_RF_tfidf = f1_score(Y_test_tfidf, Y_predict_tfidf_RF, pos_label = 'spam')
f1_logreg_word = f1_score(Y_test_tfidf, Y_predict_word_logreg, pos_label = 'spam')
f1_SVM_word = f1_score(Y_test_tfidf, Y_predict_word_SVM, pos_label = 'spam')
f1_RF_word = f1_score(Y_test_tfidf, Y_predict_word_RF, pos_label = 'spam')

# Data table
from tabulate import tabulate

data = [ 
        
        ["logistic Regression tfidf", accuracy_logreg_tfidf,precision_logreg_tfidf,
                recall_logreg_tfidf,f1_logreg_tfidf],
        ["logistic Regression word2vec", accuracy_logreg_word,precision_logreg_word,
                recall_logreg_word,f1_logreg_word],
                
        ["Naive Bayes tfidf", accuracy_NB_tfidf,precision_NB_tfidf,
                recall_NB_tfidf,f1_NB_tfidf],
              
        ["SVM tfidf", accuracy_SVM_tfidf,precision_SVM_tfidf,
                recall_SVM_tfidf,f1_SVM_tfidf],
        ["SVM word2vec", accuracy_SVM_word,precision_SVM_word,
                recall_SVM_word,f1_SVM_word],
                        
        ["Random Forest tfidf", accuracy_RF_tfidf,precision_RF_tfidf,
                recall_RF_tfidf,f1_RF_tfidf],
        ["Random Forest word2vec", accuracy_RF_word,precision_RF_word,
                recall_RF_word,f1_RF_word]
        
        ]


# Column headers
headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']

# Create and print the table
table = tabulate(data, headers, tablefmt='fancy_grid')
print(table)