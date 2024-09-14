# Import necessary libraries
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Libraries for data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Libraries for machine learning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split

# Read in the training and testing datasets using pandas
dataTraining = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)
dataTesting = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTesting.zip', encoding='UTF-8', index_col=0)

# Check for missing values in the training dataset and print out the number of missing values in each column
print(dataTraining.isnull().sum())

# Perform some data cleaning by removing rows where the 'genres', 'plot', or 'title' columns are empty or contain only whitespace
dataTraining = dataTraining[~(dataTraining['genres'].str.len() == 0)]
dataTraining = dataTraining[~dataTraining['genres'].str.isspace()]
dataTraining = dataTraining[~(dataTraining['plot'].str.len() == 0)]
dataTraining = dataTraining[~dataTraining['plot'].str.isspace()]
dataTraining = dataTraining[~(dataTraining['title'].str.len() == 0)]
dataTraining = dataTraining[~dataTraining['title'].str.isspace()]

# Transform the 'genres' column into a binary vector using the CountVectorizer class from sklearn
# This creates a new dataframe called 'genres_df' where each column represents a genre and each row contains a 1 if the movie belongs to that genre and a 0 otherwise
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
genres_dtm = vectorizer.fit_transform(dataTraining['genres'])
genres_df = pd.DataFrame(genres_dtm.toarray(), columns=vectorizer.get_feature_names_out())

# Calculate the frequency of each genre in the training dataset and create a bar plot using seaborn to visualize the results
freq_genres = genres_df.sum().sort_values(ascending=False)
plt.figure(figsize=(12, 15))
ax = sns.barplot(x=freq_genres.values, y=freq_genres.index)
ax.set(title='The most used genres', xlabel='Frecuency of use', ylabel='genres')
plt.show()

# Calculate the number of genres per movie and create a count plot using seaborn to visualize the results
genres_df['CountOfGenres'] = genres_df.sum(axis=1).sort_values(ascending=False)
genres_df['CountOfGenres'] = genres_df['CountOfGenres'].astype(str)
plt.figure(figsize=(15, 7))
ax = sns.countplot(data=genres_df, x='CountOfGenres')
ax.set(title='Frequency of Genres Number', xlabel='Number of genres', ylabel='Movies Count')
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x() + 0.3, p.get_height() + 50))
plt.show()

# Perform text preprocessing on the 'plot' column using the CountVectorizer class from sklearn
# This creates a new dataframe called 'plots_df' where each column represents a word and each row contains the frequency of that word in the movie's plot
vectorizer = CountVectorizer(analyzer='word')
plot_dtm = vectorizer.fit_transform(dataTraining['plot'])
plots_df = pd.DataFrame(plot_dtm.toarray(), columns=vectorizer.get_feature_names_out())

# Calculate the frequency of the top 100 words in the 'plots_df' dataframe and create a bar plot using seaborn to visualize the results
freq_plots = plots_df.sum().sort_values(ascending=False)[:100]
plt.figure(figsize=(12, 15))
ax = sns.barplot(x=freq_plots.values, y=freq_plots.index)
ax.set(title='The most used words', xlabel='Frecuency of Use', ylabel='Words')
plt.show()

# Print the first few rows of the testing dataset using the head() function from pandas
dataTesting.head()