import streamlit as st
import pandas as pd
import random

# --- Configuration ---
CSV_FILE = 'movies_with_clusters.csv'
CLUSTER_COLUMN = 'cluster' # The name of the cluster column in the CSV

st.set_page_config(page_title="Movie Recommendation System", layout="centered")

st.title("🎬 Movie Recommendation System (DBSCAN)")
st.write("Enter a movie name to get recommendations from the same cluster (based on similarity).")
st.markdown("---")

# Use st.cache_data to load the file once and reuse the data
@st.cache_data
def load_data():
    """Loads the movie data from the CSV."""
    try:
        df = pd.read_csv(CSV_FILE)
        # Ensure the movie title is the index for easy access, 
        # as implied by the user's original logic which samples from df.index
        if 'title' in df.columns:
             df = df.set_index('title')
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{CSV_FILE}' was not found. Please ensure it's in the same directory.")
        return pd.DataFrame() # Return empty DataFrame on error
    except KeyError:
        st.error(f"Error: The required clustering column '{CLUSTER_COLUMN}' is missing from the CSV.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return pd.DataFrame()

df_movies = load_data()

# ------------------------------------------------------------------------
# Recommendation Logic
# ------------------------------------------------------------------------

def get_recommendations(df, movie_name: str, cluster_col: str):
    """
    Finds a movie based on user input (partial match) and recommends 
    other movies from the same cluster.
    """
    if df.empty:
        return None, "Database not loaded due to error."

    # Create a column for case-insensitive search based on the index (movie titles)
    # The user's original logic relied on the index for titles
    df['name_for_search'] = df.index.str.lower()
    search_name = movie_name.lower()

    # Find the movie using partial string containment (more forgiving)
    movie = df[df['name_for_search'].str.contains(search_name, na=False)]

    if movie.empty:
        return None, f"Movie containing '{movie_name}' not found in the database."
    
    # Take the first match if multiple are found
    found_movie_title = movie.index[0]
    cluster_label = movie[cluster_col].values[0]

    if cluster_label == -1:
        return None, f"'{found_movie_title}' is classified as noise (cluster -1). Cannot provide cluster-based recommendations."

    # Filter all movies belonging to the same cluster
    cluster_movies = df[df[cluster_col] == cluster_label]
    
    # Exclude the movie itself from recommendations
    recommendable_movies = cluster_movies.index.drop(found_movie_title, errors='ignore').tolist()
    
    num_recommendations = 5
    
    if len(recommendable_movies) > num_recommendations:
        recommended_movies = random.sample(recommendable_movies, num_recommendations)
    else:
        recommended_movies = recommendable_movies
        
    if not recommended_movies:
        return found_movie_title, "No other movies found in this cluster."

    return found_movie_title, recommended_movies

# ------------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------------

movie_name = st.text_input("Enter a part of the movie title (e.g., 'Toy Story', 'Matrix')")

if st.button("Get Recommendations"):
    if not movie_name:
        st.warning("Please enter a movie name.")
    else:
        with st.spinner(f"Searching for '{movie_name}'..."):
            found_title, recommendations = get_recommendations(
                df_movies.copy(), # Pass a copy since we add a temp column
                movie_name, 
                CLUSTER_COLUMN
            )
            
            if found_title:
                st.success(f"Found: **{found_title}** (Cluster: {df_movies.loc[found_title, CLUSTER_COLUMN]})")
                st.subheader("Recommended Movies:")
                if isinstance(recommendations, list) and recommendations:
                    for i, movie in enumerate(recommendations):
                        st.markdown(f"**{i+1}.** {movie}")
                else:
                     st.info(recommendations)
            else:
                st.error(recommendations)

st.markdown("---")
st.caption(f"Note: This app uses data from `{CSV_FILE}`. Cluster labels are assumed to be in the `{CLUSTER_COLUMN}` column.")
