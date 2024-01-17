# Movie-Score-Prediction-using-HistGradientBoostingClassifier
This repository contains Python code for predicting binned IMDb scores of movies based on various features using the HistGradientBoostingClassifier from scikit-learn. The dataset used for training and testing the model is loaded from a CSV file named "movie_metadata.csv."

## Requirements

install numpy, pandas, scikit-learn

## Dataset

The dataset ("movie_metadata.csv") includes information about movies, such as duration, director and actor details, country, budget, and various other features. IMDb scores are binned into four categories: 1, 2, 3, and 4.

## Data Preprocessing

Unnecessary columns are dropped.

IMDb scores are binned into categories.

Country values are categorized, and only the top two countries are considered, while others are labeled as 'other.'

A new column is created for the sum of actor_2 and actor_3 Facebook likes.

Columns related to Facebook likes and reviews are modified.

Rows with missing values are dropped.

## Model Training

The code uses a pipeline consisting of a ColumnTransformer for preprocessing numerical and categorical features and a HistGradientBoostingClassifier for classification. The model is trained on the training dataset and evaluated on the testing dataset.

