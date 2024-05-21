# %%
import pandas as pd

train_df = pd.read_csv('data/train.csv', encoding='ISO-8859-1')
test_df = pd.read_csv('data/test.csv', encoding='ISO-8859-1')
product_descriptions_df = pd.read_csv('data/product_descriptions.csv', encoding='ISO-8859-1')
attributes_df = pd.read_csv('data/attributes.csv', encoding='ISO-8859-1')

print(train_df.head())
print(test_df.head())
print(product_descriptions_df.head())
print(attributes_df.head())

# %%
train_df = train_df.merge(product_descriptions_df, on='product_uid', how='left')
test_df = test_df.merge(product_descriptions_df, on='product_uid', how='left')

# %%
print(train_df.isnull().sum())
print(test_df.isnull().sum())

# Fill missing values in product descriptions with an empty string
train_df['product_description'].fillna('', inplace=True)
test_df['product_description'].fillna('', inplace=True)

# %%
import re
import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation and non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing
train_df['search_term'] = train_df['search_term'].apply(preprocess_text)
train_df['product_title'] = train_df['product_title'].apply(preprocess_text)
train_df['product_description'] = train_df['product_description'].apply(preprocess_text)

test_df['search_term'] = test_df['search_term'].apply(preprocess_text)
test_df['product_title'] = test_df['product_title'].apply(preprocess_text)
test_df['product_description'] = test_df['product_description'].apply(preprocess_text)

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Jaccard similarity
def jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

train_df['title_jaccard'] = train_df.apply(lambda x: jaccard_similarity(x['search_term'], x['product_title']), axis=1)
train_df['description_jaccard'] = train_df.apply(lambda x: jaccard_similarity(x['search_term'], x['product_description']), axis=1)

test_df['title_jaccard'] = test_df.apply(lambda x: jaccard_similarity(x['search_term'], x['product_title']), axis=1)
test_df['description_jaccard'] = test_df.apply(lambda x: jaccard_similarity(x['search_term'], x['product_description']), axis=1)

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


X_train = train_df[['title_jaccard', 'description_jaccard']]
y_train = train_df['relevance']
X_test = test_df[['title_jaccard', 'description_jaccard']]

# Split 
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = Ridge()

from sklearn.model_selection import GridSearchCV

# Define the grid of hyperparameters 'params'
params = {'alpha': [0.1, 0.5, 1, 2, 4]}

# Instantiate a 5-fold cross-validation grid search object 'grid'
grid = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='neg_root_mean_squared_error')

# Fit 'grid' to the training data
grid.fit(X_train_split, y_train_split)

# Extract best hyperparameters
best_hyperparams = grid.best_params_
print('Best hyperparameters:\n', best_hyperparams)

# Extract best model
best_model = grid.best_estimator_

# Predict the test set labels and evaluate the performance of the best model
y_val_pred = best_model.predict(X_val)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print(f'Validation RMSE of best model: {val_rmse}')


