
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load dataset
df = pd.read_csv("dataset/netflix_prepared.csv")

# Step 2: Load TF-IDF matrix
with open("dataset/tfidf_matrix.pkl", "rb") as file:
    tfidf_matrix = pickle.load(file)


# Step 3: Evaluation function
def evaluate_recommendations(movie_name, top_n=5):

    print("\nEVALUATING:", movie_name)

    # Find selected title
    matches = df.index[
        df["title"].str.casefold() == movie_name.casefold()
    ].tolist()

    if not matches:
        print("Title not found!")
        return

    title_index = matches[0]

    # Calculate similarity
    scores = cosine_similarity(
        tfidf_matrix[title_index],
        tfidf_matrix
    ).flatten()

    # Sort scores
    sorted_indices = scores.argsort()[::-1]

    # Remove selected title
    recommendations = [
        i for i in sorted_indices if i != title_index
    ][:top_n]

    recommended_titles = df.iloc[recommendations]["title"].tolist()

    # Evaluation checks
    print("\n1. Requested recommendations:", top_n)
    print("2. Actual recommendations:", len(recommendations))

    print(
        "3. Selected title excluded:",
        title_index not in recommendations
    )

    print(
        "4. Duplicate recommendations:",
        len(recommended_titles) != len(set(recommended_titles))
    )

    # Similarity score analysis
    recommendation_scores = scores[recommendations]

    print(
        "5. Average similarity score:",
        round(float(recommendation_scores.mean()), 3)
    )

    print("\nRECOMMENDED TITLES:")

    for index in recommendations:
        print(
            df.iloc[index]["title"],
            "|",
            round(float(scores[index]), 3)
        )


# Step 4: Test the recommendation system
evaluate_recommendations("Stranger Things")
evaluate_recommendations("The Crown")

print("\nMODEL EVALUATION COMPLETED!")
