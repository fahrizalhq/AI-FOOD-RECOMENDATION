"""Model Content-Based Filtering: TF-IDF + fitur angka, Cosine / KNN."""
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

import config

REQUIRED = ["nama", "kategori", "bahan", "cita_rasa"]


def preprocess(text) -> str:
    """Case folding, buang simbol/angka, rapikan spasi."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(path_or_buffer=None) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer or config.DATA_PATH)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ada di dataset: {missing}")
    df = df.drop_duplicates(subset="nama").dropna(subset=["nama", "bahan"])
    for c in config.NUM_COLS + ["harga"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())
    for c, default in [("vegetarian", "tidak"), ("alergi", "-")]:
        if c not in df.columns:
            df[c] = default
    return df.reset_index(drop=True)


class FoodRecommender:
    def __init__(self, df, method=None, use_numeric=None, num_weight=None,
                 knn_metric=None, include_kategori=True):
        self.df = df.reset_index(drop=True).copy()
        self.method = method or config.METHOD
        self.use_numeric = config.USE_NUMERIC if use_numeric is None else use_numeric
        num_weight = config.NUM_WEIGHT if num_weight is None else num_weight
        knn_metric = knn_metric or config.KNN_METRIC
        self.num_cols = [c for c in config.NUM_COLS if c in self.df.columns]

        # 1) Fitur teks -> TF-IDF
        text_cols = [c for c in config.TEXT_COLS
                     if c in self.df.columns and (include_kategori or c != "kategori")]
        self.df["soup"] = (self.df[text_cols].astype(str).agg(" ".join, axis=1)
                           .apply(preprocess))
        self.tfidf = TfidfVectorizer()
        X_text = self.tfidf.fit_transform(self.df["soup"])

        # 2) Fitur angka -> MinMax x bobot
        if self.use_numeric and self.num_cols:
            X_num = MinMaxScaler().fit_transform(self.df[self.num_cols]) * num_weight
            self.X = hstack([X_text, csr_matrix(X_num)]).tocsr()
        else:
            self.use_numeric = False
            self.X = X_text.tocsr()

        # 3) KNN (opsional)
        self.knn_metric = knn_metric
        if self.method == "knn":
            self.knn = NearestNeighbors(metric=knn_metric, algorithm="brute").fit(self.X)

        # Data bantu: token (untuk alasan) & set alergi (untuk filter)
        self.tokens = [set((preprocess(b) + " " + preprocess(c)).split())
                       for b, c in zip(self.df["bahan"], self.df["cita_rasa"])]
        self.alergi_sets = [set(a.strip().lower() for a in str(x).split(";")
                                if a.strip() not in ("", "-"))
                            for x in self.df["alergi"]]

    # ---------- similarity ----------
    def _scores(self, vec) -> np.ndarray:
        if self.method == "knn":
            dist, idx = self.knn.kneighbors(vec, n_neighbors=len(self.df))
            s = np.zeros(len(self.df))
            if self.knn_metric == "euclidean":
                s[idx[0]] = 1 / (1 + dist[0])
            else:
                s[idx[0]] = 1 - dist[0]
            return s
        return cosine_similarity(vec, self.X).ravel()

    def _mask(self, vegetarian_only, max_kalori, max_harga, alergi):
        m = np.ones(len(self.df), dtype=bool)
        if vegetarian_only:
            m &= (self.df["vegetarian"] == "ya").values
        if max_kalori:
            m &= (self.df["kalori"] <= max_kalori).values
        if max_harga and "harga" in self.df.columns:
            m &= (self.df["harga"] <= max_harga).values
        if alergi:
            a = {x.lower() for x in alergi}
            m &= np.array([not (s & a) for s in self.alergi_sets])
        return m

    def _format(self, scores, n, ref_tokens, ref_kategori=None, vegetarian_only=False,
                max_kalori=None, max_harga=None, alergi=None) -> pd.DataFrame:
        mask = self._mask(vegetarian_only, max_kalori, max_harga, alergi)
        rows = []
        for j in np.argsort(-scores):
            if not mask[j] or scores[j] <= 0:
                continue
            shared = sorted(ref_tokens & self.tokens[j])
            alasan = ("Kesamaan: " + ", ".join(shared[:6])) if shared else "Profil nutrisi mirip"
            if ref_kategori and self.df.at[j, "kategori"] == ref_kategori:
                alasan += f" | kategori sama ({ref_kategori})"
            row = {"idx": j, "nama": self.df.at[j, "nama"],
                   "kategori": self.df.at[j, "kategori"],
                   "cita_rasa": self.df.at[j, "cita_rasa"],
                   "skor": round(float(scores[j]), 3), "alasan": alasan}
            for c in self.num_cols + (["harga"] if "harga" in self.df.columns else []):
                row[c] = self.df.at[j, c]
            rows.append(row)
            if len(rows) == n:
                break
        return pd.DataFrame(rows)

    # ---------- API publik ----------
    def recommend(self, nama, n=None, **filters) -> pd.DataFrame:
        n = n or config.TOP_N
        match = self.df.index[self.df["nama"].str.lower() == nama.lower()]
        if len(match) == 0:
            raise ValueError(f"Makanan '{nama}' tidak ditemukan")
        i = match[0]
        s = self._scores(self.X[i])
        s[i] = -1  # jangan rekomendasikan dirinya sendiri
        return self._format(s, n, self.tokens[i], self.df.at[i, "kategori"], **filters)

    def recommend_query(self, query, n=None, **filters) -> pd.DataFrame:
        n = n or config.TOP_N
        q = preprocess(query)
        vec = self.tfidf.transform([q])
        if self.use_numeric:
            vec = hstack([vec, csr_matrix((1, len(self.num_cols)))]).tocsr()
        return self._format(self._scores(vec), n, set(q.split()), **filters)
