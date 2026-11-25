import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.metrics import silhouette_score


MODEL_NAME = "all-MiniLM-L6-v2"

def cluster_papers(df: pd.DataFrame):
    """Generates semantic embeddings, reduces dimensions, and clusters papers."""
    model = SentenceTransformer(MODEL_NAME)
    texts = (df["title"] + ". " + df["abstract"]).tolist()
    embeddings = model.encode(texts, show_progress_bar=False)
    
    # 2D UMAP projection for visualization & density clustering
    n_neighbors = min(15, max(2, len(df) - 1))
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, metric="cosine", random_state=42)
    coords = reducer.fit_transform(embeddings)
    
    min_cluster = max(3, len(df) // 15)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster, metric="euclidean")
    labels = clusterer.fit_predict(coords)
    
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df["cluster"] = [f"Cluster {c}" if c != -1 else "Outlier / Niche" for c in labels]
    
    # Calculate Silhouette Score 
    valid_mask = labels != -1
    sil_score = None
    if valid_mask.sum() > 2 and len(set(labels[valid_mask])) > 1:
        sil_score = round(float(silhouette_score(coords[valid_mask], labels[valid_mask])), 3)
        
    return df, sil_score

if __name__ == "__main__":
    from fetcher import fetch_papers

    sample_df = fetch_papers("Federated Learning Healthcare", max_results=30)
    clustered_df, score = cluster_papers(sample_df)
    print(f"Silhouette Score: {score}")
    print(clustered_df[["arxiv_id", "cluster", "x", "y"]].head(10))
