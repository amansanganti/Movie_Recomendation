from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
movies = pd.read_csv("movies.csv")

# Create features
movies["features"] = movies["genre"] + " " + movies["keywords"]

# Convert text to numbers
vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(movies["features"])

# Calculate similarity
similarity = cosine_similarity(feature_matrix)

# Recommendation function
def get_recommendations(movie_name):
    if movie_name not in movies["title"].values:
        return ["Movie not found"]

    movie_index = movies[movies["title"] == movie_name].index[0]
    scores = list(enumerate(similarity[movie_index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in sorted_scores[1:6]:
        movie_title = movies.iloc[i[0]].title
        similarity_score = round(i[1], 2)
        recommendations.append(f"{movie_title} (Similarity: {similarity_score})")

    return recommendations

# Home page
@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []

    if request.method == "POST":
        movie_name = request.form["movie"]
        recommendations = get_recommendations(movie_name)

    return render_template("index.html", recommendations=recommendations)

# Run server
if __name__ == "__main__":
    app.run(debug=True)