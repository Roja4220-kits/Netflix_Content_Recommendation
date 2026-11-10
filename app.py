
from pathlib import Path
import pickle

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Netflix | Content Recommendation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "netflix_prepared.csv"
MODEL_PATH = BASE_DIR / "dataset" / "tfidf_matrix.pkl"


# =========================================================
# 2. CUSTOM THEME
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #101014;
        color: #F5F5F5;
    }

    [data-testid="stSidebar"] {
        background-color: #18181F;
        border-right: 1px solid #30303A;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #B8B8C2;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-heading {
        font-size: 1.35rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }

    [data-testid="stMetric"] {
        background: #1C1C24;
        border: 1px solid #30303A;
        padding: 18px;
        border-radius: 12px;
    }

    div.stButton > button[kind="primary"] {
        background-color: #E50914;
        border: 1px solid #E50914;
        color: white;
        border-radius: 8px;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #B80710;
        border-color: #B80710;
        color: white;
    }

    .stProgress > div > div > div > div {
        background-color: #E50914;
    }

    hr {
        border-color: #33333D;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD DATASET AND MODEL
# =========================================================

@st.cache_data
def load_data():
    """Load the prepared Netflix dataset."""
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    """Load the saved TF-IDF matrix."""
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


try:
    df = load_data()
    tfidf_matrix = load_model()

except FileNotFoundError as error:
    st.error(f"Required project file not found: {error}")
    st.info(
        "Check that the dataset folder contains "
        "netflix_prepared.csv and tfidf_matrix.pkl."
    )
    st.stop()

except Exception as error:
    st.error(f"Unable to load the project: {error}")
    st.stop()


# Validate dataset and model alignment
if len(df) != tfidf_matrix.shape[0]:
    st.error(
        "Dataset and TF-IDF matrix do not match. "
        "Please regenerate the prepared dataset and TF-IDF matrix."
    )
    st.stop()


required_columns = [
    "title", "type", "director",
    "country", "rating", "listed_in"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Missing required dataset columns: {missing_columns}")
    st.stop()


# =========================================================
# 4. COMMON HELPER FUNCTIONS
# =========================================================

def clean_value(value):
    """Return a readable value for missing or empty metadata."""
    if pd.isna(value) or str(value).strip() == "":
        return "Not Available"

    return str(value)


def get_genre_counts():
    """Calculate the most common genres."""
    return (
        df["listed_in"]
        .fillna("")
        .str.split(", ")
        .explode()
        .loc[lambda values: values != ""]
        .value_counts()
    )


def get_country_counts():
    """Calculate the most common countries."""
    return (
        df["country"]
        .fillna("")
        .str.split(", ")
        .explode()
        .loc[lambda values: values != ""]
        .value_counts()
    )


# =========================================================
# 5. RECOMMENDATION ENGINE
# =========================================================

def generate_recommendations(selected_index, number_of_results):
    """
    Generate recommendations using cosine similarity
    between TF-IDF metadata vectors.
    """

    similarity_scores = cosine_similarity(
        tfidf_matrix[selected_index],
        tfidf_matrix
    ).flatten()

    ranked_indices = similarity_scores.argsort()[::-1]

    recommended_indices = [
        index
        for index in ranked_indices
        if index != selected_index
    ][:number_of_results]

    recommendations = []

    for index in recommended_indices:
        item = df.iloc[index]

        recommendations.append({
            "Title": clean_value(item["title"]),
            "Type": clean_value(item["type"]),
            "Genre": clean_value(item["listed_in"]),
            "Country": clean_value(item["country"]),
            "Rating": clean_value(item["rating"]),
            "Similarity": float(similarity_scores[index])
        })

    return recommendations


# =========================================================
# 6. REUSABLE DISPLAY FUNCTIONS
# =========================================================

def show_page_header(title, subtitle):
    """Display a consistent page heading."""

    st.markdown(
        f'<div class="hero-title">{title}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="hero-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


def show_recommendation_card(movie):
    """Display a recommendation using native Streamlit components."""

    with st.container(border=True):

        content_type = movie["Type"]

        if content_type == "Movie":
            st.markdown("🎬 **MOVIE**")
        else:
            st.markdown("📺 **TV SHOW**")

        st.markdown(f"### {movie['Title']}")

        st.caption(f"Genre: {movie['Genre']}")
        st.caption(f"Country: {movie['Country']}")
        st.caption(f"Rating: {movie['Rating']}")

        similarity_percentage = movie["Similarity"] * 100

        st.markdown(
            f"**Metadata Similarity: {similarity_percentage:.1f}%**"
        )

        st.progress(
            max(0.0, min(1.0, movie["Similarity"]))
        )


# =========================================================
# 7. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("# 🎬 NETFLIX")

    st.caption("AI Content Recommendation Platform")

    st.divider()

    selected_page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🎯 Recommendations",
            "📚 Explore Dataset",
            "📊 Analytics",
            "🤖 About Model"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### PROJECT DETAILS")

    st.caption("Machine Learning Project")
    st.caption("Python | Pandas | Scikit-learn")
    st.caption("TF-IDF | Cosine Similarity")


# =========================================================
# 8. DASHBOARD
# =========================================================

if selected_page == "🏠 Dashboard":

    show_page_header(
        "Discover Your Next Favourite",
        "Explore Netflix movies and TV shows with metadata-based recommendations."
    )

    total_titles = len(df)

    movie_count = (df["type"] == "Movie").sum()

    tv_show_count = (df["type"] == "TV Show").sum()

    country_count = (
        df["country"]
        .fillna("")
        .str.split(", ")
        .explode()
        .replace("", pd.NA)
        .dropna()
        .nunique()
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Total Titles", f"{total_titles:,}")
    metric2.metric("Movies", f"{movie_count:,}")
    metric3.metric("TV Shows", f"{tv_show_count:,}")
    metric4.metric("Countries", f"{country_count:,}")

    st.divider()

    st.markdown(
        '<div class="section-heading">Welcome to Netflix Recommendations</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This application recommends movies and TV shows "
        "based on similarities in their available metadata."
    )

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("### 🎯 Smart Recommendations")
            st.write(
                "Select a Netflix title and discover similar "
                "content using TF-IDF and cosine similarity."
            )

    with right:
        with st.container(border=True):
            st.markdown("### 📊 Content Analytics")
            st.write(
                "Explore the dataset through content statistics, "
                "genre distribution and country information."
            )

    st.info(
        "Choose Recommendations from the sidebar to get started."
    )


# =========================================================
# 9. RECOMMENDATIONS
# =========================================================

elif selected_page == "🎯 Recommendations":

    show_page_header(
        "🎯 Content Recommendations",
        "Choose a Netflix title and discover content with similar metadata."
    )

    title_options = df.index.tolist()

    selected_index = st.selectbox(
        "Search and select a Netflix title",
        options=title_options,
        format_func=lambda index: (
            f"{df.iloc[index]['title']} "
            f"({df.iloc[index]['type']})"
        )
    )

    control1, control2 = st.columns([3, 1])

    with control1:
        top_n = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=10,
            value=5
        )

    with control2:
        st.write("")
        st.write("")

        generate_button = st.button(
            "✨ Get Recommendations",
            type="primary",
            use_container_width=True
        )

    if generate_button:

        recommendations = generate_recommendations(
            selected_index,
            top_n
        )

        st.session_state["recommendations"] = recommendations

        st.session_state["selected_index"] = selected_index

    # Display results only for the currently selected title
    if (
        "recommendations" in st.session_state
        and st.session_state.get("selected_index") == selected_index
    ):

        selected_title = df.iloc[selected_index]

        st.divider()

        st.markdown("### Your Selected Title")

        with st.container(border=True):
            st.markdown(f"## {selected_title['title']}")
            st.caption(f"Content Type: {selected_title['type']}")
            st.caption(f"Genre: {clean_value(selected_title['listed_in'])}")

        st.markdown("### Recommended For You")

        recommendations = st.session_state["recommendations"]

        if recommendations:

            for start in range(0, len(recommendations), 2):

                columns = st.columns(2)

                for column, movie in zip(
                    columns,
                    recommendations[start:start + 2]
                ):

                    with column:
                        show_recommendation_card(movie)

        else:
            st.warning("No recommendations were found.")

        st.caption(
            "Similarity scores represent metadata similarity. "
            "They do not guarantee viewer preference or recommendation accuracy."
        )

    else:

        st.info(
            "Select a title and click Get Recommendations "
            "to view similar Netflix content."
        )


# =========================================================
# 10. EXPLORE DATASET
# =========================================================

elif selected_page == "📚 Explore Dataset":

    show_page_header(
        "📚 Explore Netflix Dataset",
        "Search and filter titles using the available dataset attributes."
    )

    filter1, filter2 = st.columns(2)

    with filter1:

        content_types = ["All"] + sorted(
            df["type"].dropna().unique().tolist()
        )

        selected_type = st.selectbox(
            "Filter by content type",
            content_types
        )

    with filter2:

        search_query = st.text_input(
            "Search title",
            placeholder="Enter a movie or TV show name..."
        )

    filtered_df = df.copy()

    if selected_type != "All":

        filtered_df = filtered_df[
            filtered_df["type"] == selected_type
        ]

    if search_query.strip():

        filtered_df = filtered_df[
            filtered_df["title"].str.contains(
                search_query,
                case=False,
                na=False,
                regex=False
            )
        ]

    st.divider()

    st.markdown(
        f"**Matching titles:** {len(filtered_df):,}"
    )

    display_columns = [
        "title",
        "type",
        "listed_in",
        "country",
        "rating"
    ]

    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 11. ANALYTICS
# =========================================================

elif selected_page == "📊 Analytics":

    show_page_header(
        "📊 Netflix Content Analytics",
        "Understand the distribution of titles across the dataset."
    )

    chart1, chart2 = st.columns(2)

    with chart1:

        st.markdown("### Content Type Distribution")

        type_counts = df["type"].value_counts()

        st.bar_chart(type_counts)

    with chart2:

        st.markdown("### Top 10 Countries")

        country_counts = get_country_counts().head(10)

        st.bar_chart(country_counts)

    st.divider()

    st.markdown("### Top 10 Genres")

    genre_counts = get_genre_counts().head(10)

    st.bar_chart(genre_counts)


# =========================================================
# 12. ABOUT THE MODEL
# =========================================================

elif selected_page == "🤖 About Model":

    show_page_header(
        "🤖 About the Recommendation Model",
        "Learn how the content-based recommendation system works."
    )

    with st.container(border=True):

        st.markdown("### 1. Data Preprocessing")

        st.write(
            "The Netflix dataset is cleaned and prepared "
            "before the recommendation process."
        )

    with st.container(border=True):

        st.markdown("### 2. Feature Engineering")

        st.write(
            "Available metadata such as content type, director, "
            "country, rating and genres are combined into "
            "a single text feature."
        )

    with st.container(border=True):

        st.markdown("### 3. TF-IDF Vectorization")

        st.write(
            "TF-IDF converts the combined text features into "
            "numerical vectors that can be compared mathematically."
        )

    with st.container(border=True):

        st.markdown("### 4. Cosine Similarity")

        st.write(
            "Cosine similarity compares the TF-IDF vectors. "
            "A higher score indicates greater similarity "
            "between the available metadata."
        )

    st.divider()

    st.markdown("### Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    tech1.markdown("**Python**")
    tech1.caption("Programming language")

    tech2.markdown("**Pandas**")
    tech2.caption("Data processing")

    tech3.markdown("**Scikit-learn**")
    tech3.caption("Machine learning")

    st.warning(
        "Model limitation: This is a content-based recommendation "
        "system. It uses available Netflix metadata and does not "
        "use viewer history, user ratings or plot descriptions."
    )


# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.caption(
    "Netflix Content Recommendation System | "
    "Built with Python, Pandas, Scikit-learn and Streamlit"
)
