"""
VERSI SEDERHANA - 1 file. Dua cara rekomendasi (Content-Based Filtering):
  (A) berdasarkan SELERA     : makanan favorit + kata kunci -> TF-IDF + Cosine/KNN
  (B) berdasarkan NUTRISI    : target kalori/protein/lemak/karbohidrat -> KNN (Euclidean)
Filter bersama: vegetarian, alergi, harga maksimum.

Jalankan:  python -m streamlit run app_sederhana.py
Butuh:     data/makanan.csv (kolom: nama, kategori, bahan, cita_rasa, kalori, protein, lemak,
                              karbohidrat, harga, vegetarian, alergi)
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

NUM = ["kalori", "protein", "lemak", "karbohidrat"]

# Contoh target nutrisi per sekali makan (kalori, protein g, lemak g, karbohidrat g).
# Ini ASUMSI untuk demo, bukan anjuran medis. Ubah sesuai kebutuhan / sumber gizi resmi.
PRESET = {
    "Diet (rendah kalori)": (300, 20, 8, 30),
    "Normal": (500, 20, 18, 60),
    "Tinggi protein": (450, 35, 15, 35),
}


# ============ 1. DATASET ============
@st.cache_data
def load_data():
    df = pd.read_csv("data/makanan.csv")
    df = df.drop_duplicates("nama").dropna(subset=["nama", "bahan"]).reset_index(drop=True)
    for c, default in [("vegetarian", "tidak"), ("alergi", "-"), ("harga", 0)]:
        if c not in df.columns:
            df[c] = default
    return df


def saring(hasil, veg=False, max_kal=None, max_harga=None, alergi=()):
    """Filter hasil: vegetarian, kalori maksimum, harga maksimum, alergen yang dihindari."""
    if veg:
        hasil = hasil[hasil["vegetarian"] == "ya"]
    if max_kal:
        hasil = hasil[hasil["kalori"] <= max_kal]
    if max_harga:
        hasil = hasil[hasil["harga"] <= max_harga]
    if alergi:
        a = {x.lower() for x in alergi}
        ok = [not ({t.strip().lower() for t in str(s).split(";")} & a) for s in hasil["alergi"]]
        hasil = hasil[ok]
    return hasil


# ============ 2A. REKOMENDASI SELERA: TF-IDF (teks) + nutrisi (angka) ============
@st.cache_resource
def fitur(use_num, w, include_kategori=True):
    df = load_data()
    teks = df["bahan"] + " " + df["cita_rasa"]
    if include_kategori:
        teks = teks + " " + df["kategori"]
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(teks.str.lower())
    if use_num:
        X = hstack([X, csr_matrix(MinMaxScaler().fit_transform(df[NUM]) * w)]).tocsr()
    return X, tfidf


def skor_vektor(vec, X, method):
    if method == "Cosine Similarity":
        return cosine_similarity(vec, X).ravel()
    dist, idx = NearestNeighbors(n_neighbors=X.shape[0]).fit(X).kneighbors(vec)
    skor = np.zeros(X.shape[0])
    skor[idx[0]] = 1 / (1 + dist[0])
    return skor


def skor_profil(X, tfidf, fav, kata, method, bobot_kata=0.5):
    """Profil selera = rata-rata vektor makanan favorit (+ vektor kata kunci)."""
    skor_fav = skor_kata = None
    if fav:
        skor_fav = skor_vektor(csr_matrix(np.asarray(X[fav].mean(axis=0))), X, method)
    if kata.strip():
        q = tfidf.transform([kata.lower()])
        q = hstack([q, csr_matrix((1, X.shape[1] - q.shape[1]))]).tocsr()
        skor_kata = skor_vektor(q, X, method)
    if skor_fav is not None and skor_kata is not None:
        return (1 - bobot_kata) * skor_fav + bobot_kata * skor_kata
    return skor_fav if skor_fav is not None else skor_kata


def top_n(df, X, skor, fav, n, **filter_):
    hasil = df.assign(skor=skor)
    if fav:
        hasil = hasil.drop(index=fav)
        mirip = cosine_similarity(X[fav], X)
        hasil["alasan"] = [df["nama"][fav[mirip[:, j].argmax()]] for j in hasil.index]
    return saring(hasil, **filter_).sort_values("skor", ascending=False).head(n)


# ============ 2B. REKOMENDASI NUTRISI: KNN jarak Euclidean ke target ============
def skor_nutrisi(df, target):
    sc = MinMaxScaler().fit(df[NUM])                    # samakan skala 0..1
    A = sc.transform(df[NUM])
    t = sc.transform(pd.DataFrame([target], columns=NUM))
    dist, idx = NearestNeighbors(n_neighbors=len(df)).fit(A).kneighbors(t)
    skor = np.zeros(len(df))
    skor[idx[0]] = 1 / (1 + dist[0])                    # makin dekat ke target, makin tinggi
    return skor


# ============ 3. PENGUJIAN & EVALUASI ============
@st.cache_data
def evaluasi(method, use_num, w, k, n_fav=2, ulang=10):
    """Rekomendasi selera: simulasi pengguna memilih 2 favorit dari 1 kategori."""
    df = load_data()
    X, _ = fitur(use_num, w, False)                     # kategori TIDAK dipakai sebagai fitur
    rng = np.random.default_rng(42)
    presisi, hit = [], []
    for kat, grup in df.groupby("kategori"):
        idx = grup.index.tolist()
        if len(idx) <= n_fav:
            continue
        for _ in range(ulang):
            fav = rng.choice(idx, n_fav, replace=False).tolist()
            skor = skor_vektor(csr_matrix(np.asarray(X[fav].mean(axis=0))), X, method)
            skor[fav] = -1
            rel = df["kategori"].values[np.argsort(-skor)[:k]] == kat
            presisi.append(rel.mean())
            hit.append(rel.any())
    return round(float(np.mean(presisi)), 3), round(float(np.mean(hit)), 3)


@st.cache_data
def evaluasi_nutrisi(k=5):
    """Rekomendasi nutrisi: seberapa dekat nutrisi Top-K ke target (MAE) dibanding semua menu."""
    df = load_data()
    baris = {}
    for nama, t in PRESET.items():
        top = np.argsort(-skor_nutrisi(df, t))[:k]
        baris[f"{nama} - Top-{k}"] = {c: round(abs(df[c].values[top] - t[i]).mean(), 1)
                                      for i, c in enumerate(NUM)}
        baris[f"{nama} - semua menu"] = {c: round(abs(df[c].values - t[i]).mean(), 1)
                                         for i, c in enumerate(NUM)}
        baris[f"{nama} - Top-{k}"]["RMSE kalori"] = round(float(np.sqrt(((df["kalori"].values[top] - t[0]) ** 2).mean())), 1)
        baris[f"{nama} - semua menu"]["RMSE kalori"] = round(float(np.sqrt(((df["kalori"].values - t[0]) ** 2).mean())), 1)
    return pd.DataFrame(baris).T


# ============ 4. INTERFACE STREAMLIT ============
st.set_page_config(page_title="Food Recommender", page_icon="🍜")
st.title("🍜 AI Food Recommendation System")
st.caption("Content-Based Filtering • TF-IDF • Cosine Similarity / KNN")

df = load_data()
rupiah = lambda x: "Rp" + f"{int(x):,}".replace(",", ".")

# ---- Filter bersama (sidebar) ----
st.sidebar.header("Filter")
veg = st.sidebar.checkbox("Vegetarian saja")
semua_alergi = sorted({a.strip() for x in df["alergi"] for a in str(x).split(";") if a.strip() not in ("", "-")})
alergi = st.sidebar.multiselect("Hindari alergen", semua_alergi)
max_harga = None
if df["harga"].max() > 0:
    h = int(df["harga"].max())
    max_harga = st.sidebar.slider("Harga maksimum (Rp)", 1000, h, h, 1000)


def tampilkan(rek):
    if rek.empty:
        st.warning("Tidak ada hasil, longgarkan filternya.")
    for _, r in rek.iterrows():
        with st.container(border=True):
            a, b = st.columns([3, 1])
            a.subheader(r["nama"])
            info = f"**{r['kategori']}** • {r['cita_rasa']} • {int(r['kalori'])} kkal"
            if r["harga"] > 0:
                info += f" • {rupiah(r['harga'])}"
            a.write(info)
            a.write(f"Protein {r['protein']:.0f} g • Lemak {r['lemak']:.0f} g • Karbohidrat {r['karbohidrat']:.0f} g")
            a.caption("Bahan: " + r["bahan"])
            if isinstance(r.get("alasan"), str):
                a.caption(f"Direkomendasikan karena mirip dengan: {r['alasan']}")
            b.metric("Kecocokan", f"{r['skor']:.2f}")


tab1, tab2, tab3, tab4 = st.tabs(["🍽️ Sesuai selera", "🥗 Sesuai nutrisi", "📊 Evaluasi & Analisis", "📋 Dataset"])

# ---- Tab 1: rekomendasi berdasarkan selera ----
with tab1:
    nama_fav = st.multiselect("Makanan yang kamu suka (boleh lebih dari satu)", sorted(df["nama"]))
    kata = st.text_input("Kata kunci selera (opsional)", placeholder="contoh: pedas ayam santan")
    c1, c2 = st.columns(2)
    method = c1.radio("Algoritma", ["Cosine Similarity", "KNN (Euclidean)"])
    n = c2.slider("Jumlah rekomendasi (Top-N)", 1, 10, 5, key="n1")
    use_num = st.checkbox("Pakai fitur nutrisi", True)
    w = st.slider("Bobot nutrisi", 0.0, 1.0, 0.5, 0.1, disabled=not use_num)
    bobot_kata = 0.5
    if nama_fav and kata.strip():
        bobot_kata = st.slider("Pengaruh kata kunci vs makanan favorit", 0.0, 1.0, 0.5, 0.1)
    max_kal = st.slider("Maksimal kalori", 50, int(df["kalori"].max()), int(df["kalori"].max()), 10)

    if st.button("Rekomendasikan", key="b1"):
        if not nama_fav and not kata.strip():
            st.warning("Pilih minimal satu makanan favorit atau isi kata kunci selera.")
        else:
            fav = [int(df.index[df["nama"] == x][0]) for x in nama_fav]
            X, tfidf = fitur(use_num, w)
            skor = skor_profil(X, tfidf, fav, kata, method, bobot_kata)
            tampilkan(top_n(df, X, skor, fav, n, veg=veg, max_kal=max_kal,
                            max_harga=max_harga, alergi=alergi))

# ---- Tab 2: rekomendasi berdasarkan kebutuhan nutrisi ----
with tab2:
    st.write("Atur target nutrisi **per sekali makan**. Sistem mencari makanan yang nutrisinya paling dekat "
             "dengan target (KNN, jarak Euclidean).")
    preset = st.selectbox("Pilih tujuan (target awal)", list(PRESET))
    t0 = PRESET[preset]
    rentang = [(int(df[c].min()), int(df[c].max())) for c in NUM]
    kal = st.slider("Target kalori (kkal)", rentang[0][0], rentang[0][1], min(max(t0[0], rentang[0][0]), rentang[0][1]), 10, key=f"kal{preset}")
    pro = st.slider("Target protein (g)", rentang[1][0], rentang[1][1], min(max(t0[1], rentang[1][0]), rentang[1][1]), 1, key=f"pro{preset}")
    lem = st.slider("Target lemak (g)", rentang[2][0], rentang[2][1], min(max(t0[2], rentang[2][0]), rentang[2][1]), 1, key=f"lem{preset}")
    kar = st.slider("Target karbohidrat (g)", rentang[3][0], rentang[3][1], min(max(t0[3], rentang[3][0]), rentang[3][1]), 1, key=f"kar{preset}")
    n2 = st.slider("Jumlah rekomendasi (Top-N)", 1, 10, 5, key="n2")
    st.caption("Target awal hanya contoh/asumsi demo, bukan anjuran medis.")

    if st.button("Cari makanan sesuai nutrisi", key="b2"):
        skor = skor_nutrisi(df, (kal, pro, lem, kar))
        tampilkan(saring(df.assign(skor=skor), veg=veg, max_harga=max_harga, alergi=alergi)
                  .sort_values("skor", ascending=False).head(n2))

# ---- Tab 3: evaluasi + analisis + perbaikan ----
with tab3:
    k = st.select_slider("K (jumlah rekomendasi yang dinilai)", [3, 5, 10], value=5)

    st.subheader("A. Rekomendasi selera: perbandingan model")
    konfigurasi = {
        "TF-IDF + Cosine": ("Cosine Similarity", False, 0.0),
        "TF-IDF + Nutrisi + Cosine": ("Cosine Similarity", True, 0.5),
        "TF-IDF + KNN (Euclidean)": ("KNN (Euclidean)", False, 0.0),
        "TF-IDF + Nutrisi + KNN (Euclidean)": ("KNN (Euclidean)", True, 0.5),
    }
    hasil = {nama: evaluasi(m, u, w_, k) for nama, (m, u, w_) in konfigurasi.items()}
    tabel = pd.DataFrame(hasil, index=[f"Precision@{k}", "Hit Rate"]).T
    st.dataframe(tabel, use_container_width=True)
    st.bar_chart(tabel[[f"Precision@{k}"]])
    terbaik = tabel[f"Precision@{k}"].idxmax()
    st.success(f"Model terbaik pada K={k}: **{terbaik}** (Precision@{k} = {tabel.loc[terbaik].iloc[0]})")

    st.subheader("Perbaikan: pengaruh bobot nutrisi (Cosine)")
    sweep = pd.Series({w_: evaluasi("Cosine Similarity", True, w_, k)[0] for w_ in [0.0, 0.25, 0.5, 0.75, 1.0]},
                      name=f"Precision@{k}")
    st.line_chart(sweep)
    st.caption("Simulasi: pengguna memilih 2 favorit dari 1 kategori. Relevan = rekomendasi sekategori. "
               "Hit Rate = proporsi simulasi dengan minimal 1 rekomendasi relevan.")

    st.subheader("B. Rekomendasi nutrisi: selisih terhadap target")
    st.dataframe(evaluasi_nutrisi(k), use_container_width=True)
    st.caption("Angka = rata-rata selisih absolut (MAE) terhadap target. Makin kecil makin sesuai. "
               "Baris 'semua menu' adalah pembanding (tanpa sistem rekomendasi).")

# ---- Tab 4: dataset ----
with tab4:
    st.dataframe(df, use_container_width=True)
    st.caption(f"{len(df)} menu")
