import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import classification_report

# Load data
movie_df = pd.read_csv("movie_metadata.csv")

# Dropping unnecessary columns
movie_df.drop(['movie_imdb_link', 'color', 'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name',
               'movie_title', 'plot_keywords', 'genres'], axis=1, inplace=True)

# Binning IMDb scores
movie_df["imdb_binned_score"] = pd.cut(movie_df['imdb_score'], bins=[0, 4, 6, 8, 10], right=True, labels=False) + 1

# Dropping IMDb score column
movie_df.drop('imdb_score', axis=1, inplace=True)

# Categorizing countries
value_counts = movie_df["country"].value_counts()
top_countries = value_counts[:2].index
movie_df['country'] = movie_df.country.where(movie_df.country.isin(top_countries), 'other')

# Creating a new column for the sum of actor_2 and actor_3 Facebook likes
movie_df['Other_actor_facebook_likes'] = movie_df["actor_2_facebook_likes"] + movie_df['actor_3_facebook_likes']

# Dropping unnecessary columns
movie_df.drop(['actor_2_facebook_likes', 'actor_3_facebook_likes', 'cast_total_facebook_likes'], axis=1, inplace=True)
movie_df.drop_duplicates(inplace=True)

# Ratio of critic reviews to user reviews
movie_df['critic_review_ratio'] = movie_df['num_critic_for_reviews'] / movie_df['num_user_for_reviews']

# Dropping columns
movie_df.drop(['num_critic_for_reviews', 'num_user_for_reviews'], axis=1, inplace=True)

# Dropping rows with missing values
movie_df.dropna(inplace=True)

# Splitting data into features and target
X = movie_df.drop('imdb_binned_score', axis=1)
y = movie_df['imdb_binned_score']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Defining numerical and categorical features
numerical_features = ['duration', 'director_facebook_likes', 'actor_1_facebook_likes', 'gross',
                       'num_voted_users', 'facenumber_in_poster', 'budget', 'title_year', 'aspect_ratio',
                       'movie_facebook_likes', 'Other_actor_facebook_likes', 'critic_review_ratio']
categorical_features = ['country', 'content_rating']

# Creating a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features),
    ])

# Creating a pipeline for the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(max_iter=100, learning_rate=0.09, max_depth=5))
])

# Training the model
pipeline.fit(X_train, y_train)

# Making predictions
y_pred = pipeline.predict(X_test)

# Evaluating the model
print("Accuracy of HistGradientBoostingClassifier with dropping missing values:", metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


