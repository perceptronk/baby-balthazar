import logging
import zipfile

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from model import OverviewRecommender
from sentence_transformers import SentenceTransformer, util
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--embed', action='store_true')

    args = parser.parse_args()
    
    if args.embed:
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
        overview_cos_sim = util.cos_sim(overview_embeddings, overview_embeddings).half()

        # Save the lower triangular part of the cosine similarity matrix
        torch.save(
            overview_cos_sim[torch.tril(torch.ones_like(overview_cos_sim, dtype=torch.bool))],
            './onnx/overview_cos_sim.pt'
        )

    # Construct the export model
    recommender = OverviewRecommender('./data/tmdb_5000_movies.csv', './onnx/overview_cos_sim.pt')

    # Convert to onnx
    torch.onnx.export(
        recommender,
        recommender.encode_str('The Dark Knight', recommender.max_len), 
        './onnx/overview_recommender.onnx', 
        input_names=['movie_title'], 
        output_names=['titles', 'scores'],
        export_params=True,
        do_constant_folding=True
    )

    # Test the onnx model
    ort_session = ort.InferenceSession('./onnx/overview_recommender.onnx')
    movie_title = recommender.encode_str('The Dark Knight', ort_session.get_inputs()[0].shape[0]).numpy()
    outputs = ort_session.run(None, {'movie_title': movie_title})
    print(dict(zip(map(recommender.decode_str, torch.from_numpy(outputs[0])), map(float, outputs[1]))))

    for inp in ort_session.get_inputs():
        print(f'{inp.name = } {inp.shape = } {inp.type = }')
    
    for out in ort_session.get_outputs():
        print(f'{out.name = } {out.shape = } {out.type = }')