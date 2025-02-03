import torch
import torch.nn as nn
import pandas as pd

class OverviewRecommender(nn.Module):
    def __init__(self, movies_path, cos_sim_path, cos_sim_dim=4800, cos_sim_dtype=torch.float16):
        super(OverviewRecommender, self).__init__()
        movie = pd.read_csv(movies_path)
        
        # Drop rows where overview is null
        movie = movie.dropna(subset=['overview'])
        movie = movie.reset_index(drop=True)
        
        # Encode titles as bytes
        original_titles = movie['original_title'].values
        max_len = max(map(lambda s: len(s.encode('utf-8')), original_titles))
        original_titles = torch.stack([self.encode_str(s, max_len) for s in original_titles])
        self.register_buffer('original_titles', original_titles)
        self.register_buffer('max_len', torch.tensor(max_len, dtype=torch.int32))

        # Load cosine similarity matrix tril and decode to full matrix
        overview_cos_sim_tril = torch.load(cos_sim_path, weights_only=True)
        overview_cos_sim = torch.zeros(cos_sim_dim, cos_sim_dim, dtype=cos_sim_dtype)
        idx = torch.tril_indices(cos_sim_dim, cos_sim_dim)
        overview_cos_sim[idx[0], idx[1]] = overview_cos_sim_tril
        overview_cos_sim = torch.maximum(overview_cos_sim, overview_cos_sim.T)
        self.register_buffer('overview_cos_sim', overview_cos_sim)

    def encode_str(self, s, max_len):
        # Convert string to bytes and pad to max_len
        s = s.encode('utf-8')
        s = s[:max_len] + b'\0' * (max_len - len(s))
        return torch.tensor(list(s), dtype=torch.int)

    def decode_str(self, t):
        # Convert bytes to string and remove padding
        return t.to(torch.int8).numpy().tobytes().rstrip(b'\0').decode('utf-8')

    def forward(self, movie_title):
        scores = self.overview_cos_sim[(self.original_titles == movie_title).all(-1)].squeeze()
        top = torch.topk(scores, 11)

        # Remove self-similarity and convert to list
        top_scores = top.values[1:]
        top_indices = top.indices[1:]

        titles = self.original_titles[top_indices]
        return titles, top_scores
