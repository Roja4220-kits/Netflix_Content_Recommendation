# 🎬 Netflix Content Recommendation System

A content-based recommendation system that suggests similar Netflix movies and TV shows using Machine Learning techniques.

## 📌 Project Overview

The Netflix Content Recommendation System helps users discover movies and TV shows similar to their selected titles.

It uses metadata such as content type, director, country, rating, and genres to identify similarities between Netflix titles.

The recommendation engine uses **TF-IDF Vectorization** and **Cosine Similarity** to generate recommendations.

## 🎯 Project Objectives

- Analyze and preprocess the Netflix dataset.
- Perform Exploratory Data Analysis (EDA).
- Extract useful metadata features.
- Convert text data into numerical vectors using TF-IDF.
- Calculate similarities using Cosine Similarity.
- Recommend similar movies and TV shows.
- Build an interactive web application using Streamlit.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine Learning |
| TF-IDF | Text vectorization |
| Cosine Similarity | Similarity calculation |
| Streamlit | Web application |

## 📂 Project Structure

```text
Netflix_Content_Recommendation/
│
├── dataset/
│   ├── netflix_titles.csv
│   ├── netflix_cleaned.csv
│   ├── netflix_prepared.csv
│   ├── tfidf_matrix.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── tfidf_vectorization.py
│   ├── cosine_similarity.py
│   ├── recommendation_engine.py
│   └── model_evaluation.py
│
├── screenshots/
│   ├── content_type.png
│   ├── top_countries.png
│   ├── top_genres.png
│   ├── release_year.png
│   ├── dashboard.png
│   ├── recommendations.png
│   ├── analytics.png
│   └── about_model.png
│
├── app.py
├── requirements.txt
└── README.md
```

## 📊 Dataset Information

The project uses the Netflix Titles dataset.

- Original records: 8,790
- Records after cleaning: 8,789
- Original columns: 10

The dataset contains information such as:

- Show ID
- Content Type
- Title
- Director
- Country
- Date Added
- Release Year
- Rating
- Duration
- Listed Genres

## ⚙️ How the Recommendation System Works

### 1. Data Preprocessing

The dataset is loaded, inspected, cleaned, and prepared for analysis.

### 2. Exploratory Data Analysis

Analyze content types, countries, genres, and release-year distributions.

### 3. Feature Engineering

Combine available metadata fields into a single text feature.

### 4. TF-IDF Vectorization

Convert text features into numerical vectors using TF-IDF.

### 5. Cosine Similarity

Calculate similarity between the TF-IDF vectors.

### 6. Recommendation Generation

Rank titles based on their similarity scores and return the requested number of recommendations.

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project Folder

```bash
cd Netflix_Content_Recommendation
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
python -m streamlit run app.py
```

### Step 5: Open the Application

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 🌐 Application Features

- Interactive dashboard with dataset statistics.
- Movie and TV show recommendations.
- Searchable Netflix title selection.
- Adjustable recommendation count.
- Recommendation similarity scores.
- Dataset exploration and filtering.
- Content analytics and visualizations.
- Explanation of the recommendation methodology.

## ⚠️ Model Limitations

- Recommendations are based on available metadata.
- The system does not use user viewing history.
- It does not use viewer ratings or plot descriptions.
- Similarity scores represent metadata similarity, not guaranteed recommendation accuracy.
- The system is not a collaborative filtering model.

## 🔮 Future Improvements

- Integrate movie poster images.
- Include richer content descriptions and cast information when available.
- Improve feature weighting and recommendation quality.
- Evaluate recommendations using Precision@K and other ranking metrics.
- Deploy the application online.

## 👩‍💻 Author

**Roja Pasupuleti**

Machine Learning Intern Project

**Technologies:** Python | Pandas | Scikit-learn | Streamlit

---

*This project was developed as part of a Machine Learning Internship task.*