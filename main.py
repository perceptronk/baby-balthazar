from fastapi import FastAPI, Response
from pydantic import BaseModel
import logging
import zipfile
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os

app = FastAPI(title="Baby Balthazar")
logging.basicConfig(level = logging.INFO)

# Unzip and load the data
with zipfile.ZipFile('./data/tmdb_5000_movies.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('./data')

movie = pd.read_csv("./data/tmdb_5000_movies.csv")

# Drop rows where overview is null
movie = movie.dropna(subset=['overview'])
movie = movie.reset_index(drop=True)

# Load the embedding model
model = SentenceTransformer('intfloat/multilingual-e5-base')

# Compute embeddings for all movie overviews
overview_embeddings = model.encode(movie['overview'])

# Compute cosine similarity between all pairs
overview_cos_sim = util.cos_sim(overview_embeddings, overview_embeddings)

class DataInput(BaseModel):
    movie: str

class SimilarMovie(BaseModel):
    movie: str
    score: float

class Response(BaseModel):
    similar: list[SimilarMovie]

@app.post("/recommend")
def recommend(input: DataInput): 
    result = pd.concat([movie["original_title"], 
                    pd.DataFrame(overview_cos_sim[:,movie[movie["original_title"] == input.movie].index].numpy(), columns=['score'])],axis = 1)
    result = result[~np.isclose(result['score'], 1.0, atol=1e-6)]
    result = result.sort_values('score', ascending= False).head(10).reset_index(drop =  True)
    result = result.rename(columns={"original_title": "movie"})
    return Response(similar=result.to_dict(orient='records'))

@app.get("/", include_in_schema=False)
async def root():
    return "Ok"

os.system("say 'Baby Balthazar is up and running!'") 

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)