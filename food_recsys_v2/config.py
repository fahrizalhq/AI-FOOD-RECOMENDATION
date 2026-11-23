"""
PUSAT PENGATURAN — ubah nilai di sini untuk bereksperimen.
Semua file lain (recommender.py, evaluate.py, app.py) membaca dari sini.
"""

# --- Data ---
DATA_PATH = "data/makanan.csv"

# Kolom teks yang digabung untuk TF-IDF (urutan tidak penting)
TEXT_COLS = ["bahan", "cita_rasa", "kategori"]

# Kolom angka yang ikut dihitung kemiripannya (hapus/tambah sesuai dataset)
NUM_COLS = ["kalori", "protein", "lemak", "karbohidrat"]

# --- Model ---
METHOD = "cosine"        # "cosine" atau "knn"
KNN_METRIC = "cosine"    # "cosine" atau "euclidean" (hanya untuk knn)
USE_NUMERIC = True       # False = hanya pakai teks (TF-IDF)
NUM_WEIGHT = 0.5         # bobot fitur angka: 0 (abaikan) sampai 1 (kuat)

# --- Rekomendasi ---
TOP_N = 5

# --- Evaluasi ---
EVAL_K = [3, 5, 10]                     # nilai K untuk Precision@K
WEIGHT_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]  # bobot yang diuji di eksperimen
