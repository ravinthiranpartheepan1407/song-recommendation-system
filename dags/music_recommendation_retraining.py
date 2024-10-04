from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import re
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
from mlflow.exceptions import MlflowException

# Define helper functions
def parse_genres(genre_string):
    return re.findall(r"'(.*?)'", genre_string)

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

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 4),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'music_recommendation_system',
    default_args=default_args,
    description='A DAG to train and log a music recommendation system',
    schedule_interval=timedelta(days=1),
    catchup=False
)

# Define functions for each step of the process
def load_and_preprocess_data(**kwargs):
    df = pd.read_csv("/opt/airflow/datasets/tracks.csv")
    df['artist_genres'] = df['artist_genres'].apply(parse_genres)
    df_dict = df.to_dict(orient='records')
    kwargs['ti'].xcom_push(key='preprocessed_data', value=df_dict)

def create_user_genre_matrix_task(**kwargs):
    ti = kwargs['ti']
    df_dict = ti.xcom_pull(key='preprocessed_data', task_ids='load_and_preprocess_data')
    df = pd.DataFrame(df_dict)
    all_genres = set([genre for genres in df['artist_genres'] for genre in genres])
    user_genre_matrix = create_user_genre_matrix(df, all_genres)
    user_genre_matrix_dict = user_genre_matrix.to_dict(orient='split')
    ti.xcom_push(key='user_genre_matrix', value=user_genre_matrix_dict)

def split_data(**kwargs):
    ti = kwargs['ti']
    user_genre_matrix_dict = ti.xcom_pull(key='user_genre_matrix', task_ids='create_user_genre_matrix')
    user_genre_matrix = pd.DataFrame(**user_genre_matrix_dict)
    train_matrix, test_matrix = train_test_split(user_genre_matrix, test_size=0.2, random_state=42)
    ti.xcom_push(key='train_matrix', value=train_matrix.to_dict(orient='split'))
    ti.xcom_push(key='test_matrix', value=test_matrix.to_dict(orient='split'))

def create_mlflow_experiment_if_not_exists(experiment_name):
    mlflow.set_tracking_uri("http://mlflow:5000")
    try:
        mlflow.create_experiment(experiment_name)
    except MlflowException:
        print(f"Experiment '{experiment_name}' already exists.")

def train_and_evaluate_model(**kwargs):
    ti = kwargs['ti']
    train_matrix_dict = ti.xcom_pull(key='train_matrix', task_ids='split_data')
    test_matrix_dict = ti.xcom_pull(key='test_matrix', task_ids='split_data')
    train_matrix = pd.DataFrame(**train_matrix_dict)
    test_matrix = pd.DataFrame(**test_matrix_dict)

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Music_Recommendation_System")

    with mlflow.start_run():
        # Create and train the model
        model = CollaborativeFilteringModel(train_matrix)
        model.fit(train_matrix)

        # Make predictions
        predictions = model.predict(test_matrix)

        # Calculate MSE
        mse = mean_squared_error(test_matrix.values, predictions)
        print(f"Mean Squared Error: {mse}")

        # Log metrics
        mlflow.log_metric("mean_squared_error", mse)

        # Log parameters
        mlflow.log_param("n_similar_users", model.n_similar_users)

        # Save the model
        model_path = "/opt/airflow/models/collaborative_filtering_model.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Log the model file to MLflow
        mlflow.log_artifact(model_path)

# Create tasks
load_data_task = PythonOperator(
    task_id='load_and_preprocess_data',
    python_callable=load_and_preprocess_data,
    provide_context=True,
    dag=dag,
)

create_matrix_task = PythonOperator(
    task_id='create_user_genre_matrix',
    python_callable=create_user_genre_matrix_task,
    provide_context=True,
    dag=dag,
)

split_data_task = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    provide_context=True,
    dag=dag,
)

create_experiment_task = PythonOperator(
    task_id='create_mlflow_experiment',
    python_callable=create_mlflow_experiment_if_not_exists,
    op_kwargs={'experiment_name': 'Music_Recommendation_System'},
    dag=dag,
)

train_and_evaluate_task = PythonOperator(
    task_id='train_and_evaluate_model',
    python_callable=train_and_evaluate_model,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
load_data_task >> create_matrix_task >> split_data_task >> create_experiment_task >> train_and_evaluate_task