"""Evaluasi: Precision@K, Coverage, Diversity (leave-one-out)."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import config
from recommender import FoodRecommender, load_data


def evaluate(df, k=5, **model_kwargs) -> dict:
    model = FoodRecommender(df, **model_kwargs)
    precisions, diversities, seen = [], [], set()
    for i in range(len(df)):
        recs = model.recommend(df.at[i, "nama"], n=k)
        if recs.empty:
            continue
        precisions.append((recs["kategori"] == df.at[i, "kategori"]).mean())
        seen.update(recs["idx"])
        sim = cosine_similarity(model.X[recs["idx"].tolist()])
        iu = np.triu_indices_from(sim, k=1)
        diversities.append(1 - sim[iu].mean() if len(iu[0]) else 0)
    return {f"Precision@{k}": round(float(np.mean(precisions)), 3),
            "Coverage": round(len(seen) / len(df), 3),
            "Diversity": round(float(np.mean(diversities)), 3)}


def compare_models(df, k=5) -> pd.DataFrame:
    """Bandingkan beberapa konfigurasi. Tambah baris baru di dict untuk eksperimen sendiri."""
    configs = {
        "TF-IDF + Cosine": dict(method="cosine", use_numeric=False),
        "TF-IDF + KNN (cosine)": dict(method="knn", use_numeric=False, knn_metric="cosine"),
        "TF-IDF + Nutrisi + Cosine": dict(method="cosine", use_numeric=True),
        "TF-IDF + Nutrisi + KNN (cosine)": dict(method="knn", use_numeric=True, knn_metric="cosine"),
        "TF-IDF + Nutrisi + KNN (euclidean)": dict(method="knn", use_numeric=True, knn_metric="euclidean"),
    }
    # include_kategori=False agar label evaluasi tidak bocor ke fitur
    return pd.DataFrame({name: evaluate(df, k, include_kategori=False, **cfg)
                         for name, cfg in configs.items()}).T


def sweep_weight(df, k=5) -> pd.DataFrame:
    """Uji beberapa bobot fitur nutrisi (bahan tabel eksperimen di laporan)."""
    rows = {f"bobot {w}": evaluate(df, k, include_kategori=False, method="cosine",
                                   use_numeric=True, num_weight=w)
            for w in config.WEIGHT_SWEEP}
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    data = load_data()
    for k in config.EVAL_K:
        print(f"\n=== Perbandingan model, K = {k} ===")
        print(compare_models(data, k))
    print("\n=== Eksperimen bobot nutrisi, K = 5 ===")
    res = sweep_weight(data, 5)
    print(res)
    res.to_csv("hasil_eksperimen_bobot.csv")
    print("\nTersimpan: hasil_eksperimen_bobot.csv")
