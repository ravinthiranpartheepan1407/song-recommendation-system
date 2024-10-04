import streamlit as st
import pandas as pd
import dill as pickle  # Use dill for better serialization
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_model(model_path):
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model_path = "models/collaborative_filtering_model.pkl"
model = load_model(model_path)

@st.cache_data
def load_song_data():
    song_data_path = "./data/tracks.csv"
    return pd.read_csv(song_data_path)

song_data = load_song_data()

@st.cache_resource
def create_fallback_recommender():
    song_data['all_genres'] = song_data['artist_genres'].apply(eval).apply(' '.join)
    tfidf = TfidfVectorizer(stop_words='english')
    genre_matrix = tfidf.fit_transform(song_data['all_genres'])
    return tfidf, genre_matrix

tfidf, genre_matrix = create_fallback_recommender()

@st.cache_data
def get_unique_genres():
    all_genres = set()
    for genres in song_data['artist_genres'].apply(eval):
        all_genres.update(genres)
    return sorted(list(all_genres))

unique_genres = get_unique_genres()

st.title("Music Recommendation System")

selected_genres = st.multiselect(
    "Select Your Preferred Genres:",
    options=unique_genres,
    default=["pop"]
)

if st.button("Get Recommendations"):
    if not selected_genres:
        st.warning("Please select at least one genre.")
    else:
        # Combine selected genres
        combined_genre = " ".join(selected_genres).lower()

        if model is not None and all(genre.lower() in model.user_genre_matrix.columns for genre in selected_genres):
            # Use the original model if available
            user_profile = np.zeros(len(model.user_genre_matrix.columns))
            for genre in selected_genres:
                user_profile[model.user_genre_matrix.columns.get_loc(genre.lower())] = 100
            user_profile = user_profile.reshape(1, -1)
            recommendations = model.predict(pd.DataFrame(user_profile, columns=model.user_genre_matrix.columns))
        else:
            # Fallback to TF-IDF based recommendation
            user_profile = tfidf.transform([combined_genre])
            similarity_scores = cosine_similarity(user_profile, genre_matrix).flatten()
            top_indices = similarity_scores.argsort()[-5:][::-1]
            recommendations = song_data.iloc[top_indices]

        songs_in_genre = song_data[song_data['artist_genres'].apply(lambda x: any(genre.lower() in [g.lower() for g in eval(x)] for genre in selected_genres))]

        if not songs_in_genre.empty:
            # Get the top 5 songs in the specified genres by track popularity
            top_songs = songs_in_genre.nlargest(5, 'track_pop')
            recommended_songs = []

            for _, song in top_songs.iterrows():
                recommended_songs.append({
                    "name": song['names'],
                    "artist": song['artist_names'],
                    "track_uri": song['uris'],
                })

            st.subheader(f"Top 5 Recommended Songs for Genres: {', '.join(selected_genres)}")
            for song in recommended_songs:
                st.write(f"**{song['name']}** by *{song['artist']}*")
                st.write(f"[Listen Here]({song['track_uri']})")  # Link to listen to the song
        else:
            st.write(f"No songs found for the selected genres: {', '.join(selected_genres)}")

st.write("Developed by Ravinthiran - Music Recommendation System")