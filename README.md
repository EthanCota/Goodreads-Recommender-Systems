# Goodreads-Recommender-Systems
A data analysis project that impliments recommender systems on a dataset of Goodreads data to predict user-book interactions

# Dataset
The dataset used is a collection of anonymized user-book review interactions sourced from Goodreads

# Modeling
## Ratings Prediction
For one model, the goal is to predict the expected rating from a user-book pair. This is implimented using a Latent Factor Model.
## Read Status Prediction
For the second model, the goal is to predict whether or not a user-book pair exists (user has read book).
This is implimented using a heuristic-based recommender system, which relies on a combination of cosine similarity and the overall popularity of the book we are predicting.