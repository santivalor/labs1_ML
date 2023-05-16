from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
import requests
import io


#df_preliminar = pd.read_csv('/Users/melisatirabassi/Library/CloudStorage/OneDrive-Personal/Documentos/Diplomatura Data Science/Henry/dataset_ml.csv')
#similarity_matrix = joblib.load('modelo.joblib')

app = FastAPI()

def download_file_from_dropbox(file_path):
    try:
        response = requests.get(f"https://www.dropbox.com/s/{file_path}?dl=1")
        response.raise_for_status()
        file_data = response.content
        return file_data
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from Dropbox: {e}")


# Ejemplo de uso
file_path = "26kg975cdm37zcn/dataset_ml.csv"  # Ruta del archivo en Dropbox
file_path_model = "brxv5z0wda1myny/matriz_similitud.pkl"  # Ruta del archivo en Dropbox
file_data_df = download_file_from_dropbox(file_path)
file_data_model = download_file_from_dropbox(file_path_model)
# Procesa el archivo de datos según tus necesidades

df_preliminar= pd.read_csv(io.BytesIO(file_data_df))
similarity_matrix = joblib.load(io.BytesIO(file_data_model))

@app.post("/peliculas_recomendadas")
def predict(data: str):
    # Make predictions using the loaded model
    input_movie_index = df_preliminar.index[df_preliminar['title']==data]
    movie_scores = similarity_matrix[input_movie_index]
    similar_movie_indices = np.argsort(movie_scores)
    similar_movie_indices = similar_movie_indices[-1][::-1]
    similar_movie_indices = similar_movie_indices[:6]
    recommended_movies = []
    for index in similar_movie_indices:
        if index < len(df_preliminar):
            recommended_movies.append(df_preliminar.iloc[index]['title'])
    # Return the predictions as the API response
    return {"predictions": recommended_movies}

