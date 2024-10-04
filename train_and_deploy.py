import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import re
import dill as pickle

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Song Recommendation Model Auto Training")

df = pd.read_csv("./data/tracks.csv")


def parse_genres(genre_string):
    return re.findall(r"'(.*?)'", genre_string)


df['artist_genres'] = df['artist_genres'].apply(parse_genres)

all_genres = set([genre for genres in df['artist_genres'] for genre in genres])


def create_user_genre_matrix(df, all_genres, n_users=100):
    user_genre_data = []
    for user_id in range(n_users):
        for genre in all_genres:
            tracks_with_genre = df[df['artist_genres'].apply(lambda x: genre in x)]
            if not tracks_with_genre.empty:
                interaction = np.random.choice(tracks_with_genre['track_pop'])
                user_genre_data.append([user_id, genre, interaction])

    user_genre_df = pd.DataFrame(user_genre_data, columns=['user_id', 'genre', 'interaction'])
    return user_genre_df.pivot(index='user_id', columns='genre', values='interaction').fillna(0)


class CollaborativeFilteringModel:
    def __init__(self, user_genre_matrix, n_similar_users=5):
        self.user_genre_matrix = user_genre_matrix
        self.n_similar_users = n_similar_users

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        recommendations = []
        for _, user_profile in X.iterrows():
            similarity = cosine_similarity(user_profile.values.reshape(1, -1), self.user_genre_matrix)[0]
            similar_users = similarity.argsort()[::-1][:self.n_similar_users]
            user_recommendations = self.user_genre_matrix.iloc[similar_users].mean()
            recommendations.append(user_recommendations)
        return np.array(recommendations)


# Main execution
if __name__ == "__main__":
    with mlflow.start_run():
        user_genre_matrix = create_user_genre_matrix(df, all_genres)
        train_matrix, test_matrix = train_test_split(user_genre_matrix, test_size=0.2, random_state=42)

        model = CollaborativeFilteringModel(train_matrix)
        model.fit(train_matrix)

        predictions = model.predict(test_matrix)

        mse = mean_squared_error(test_matrix.values, predictions)
        print(f"Mean Squared Error: {mse}")

        mlflow.log_metric("mean_squared_error", mse)

        mlflow.log_param("n_similar_users", model.n_similar_users)

        with open("models/collaborative_filtering_model.pkl", "wb") as f:
            pickle.dump(model, f)

        mlflow.log_artifact("models/collaborative_filtering_model.pkl")
