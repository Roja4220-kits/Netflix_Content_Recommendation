
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Step 1: Load prepared dataset
df = pd.read_csv("dataset/netflix_prepared.csv")

# Step 2: Handle empty values
df["combined_features"] = (
    df["combined_features"].fillna("").astype(str)
)

# Step 3: Create TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words="english")

# Step 4: Convert text into numerical vectors
tfidf_matrix = tfidf.fit_transform(df["combined_features"])

# Step 5: Display results
print("\nTF-IDF MATRIX SHAPE:")
print(tfidf_matrix.shape)

print("\nNUMBER OF UNIQUE WORDS:")
print(len(tfidf.get_feature_names_out()))

print("\nFIRST 10 WORDS:")
print(tfidf.get_feature_names_out()[:10])

print("\nFIRST TITLE VECTOR:")
print(tfidf_matrix[0].toarray())

# Step 6: Save the TF-IDF matrix
with open("dataset/tfidf_matrix.pkl", "wb") as file:
    pickle.dump(tfidf_matrix, file)

# Step 7: Save the vectorizer
with open("dataset/tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(tfidf, file)

print("\nTF-IDF VECTORIZATION COMPLETED SUCCESSFULLY!")
