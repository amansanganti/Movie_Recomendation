import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie dataset
movies = pd.read_csv("movies.csv")

# Combine genre and keywords into one feature
movies["features"] = movies["genre"] + " " + movies["keywords"]

# Convert text data into numbers (machine learning step)
vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(movies["features"])

# Calculate similarity between movies
similarity = cosine_similarity(feature_matrix)

# Function to recommend movies
def recommend(movie_name):
    # Check if movie exists
    if movie_name not in movies["title"].values:
        print("\n❌ Movie not found in database.")
        print("Try one of these:")
        print(movies["title"].tolist())
        return

    movie_index = movies[movies["title"] == movie_name].index[0]
    scores = list(enumerate(similarity[movie_index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended Movies:")
    for i in sorted_scores[1:6]:
        movie_title = movies.iloc[i[0]].title
        similarity_score = i[1]
        print(f"{movie_title} (Similarity: {similarity_score:.2f})")

# User input
movie = input("Enter a movie name which you have recently watched: ")
recommend(movie)