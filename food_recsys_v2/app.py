import streamlit as st

import config
from recommender import FoodRecommender, load_data
from evaluate import compare_models, sweep_weight

st.set_page_config(page_title="AI Food Recommender", page_icon="🍜", layout="wide")
st.title("🍜 AI Food Recommendation System")
st.caption("Content-Based Filtering • TF-IDF • Cosine Similarity / KNN")

# ---------- Data ----------
upload = st.sidebar.file_uploader("Upload dataset CSV (opsional)", type="csv")
df = load_data(upload) if upload else load_data()

# ---------- Pengaturan model ----------
st.sidebar.header("Pengaturan Model")
method = st.sidebar.radio("Algoritma", ["cosine", "knn"],
                          index=["cosine", "knn"].index(config.METHOD),
                          format_func=lambda x: {"cosine": "Cosine Similarity", "knn": "K-Nearest Neighbors"}[x])
knn_metric = config.KNN_METRIC
if method == "knn":
    knn_metric = st.sidebar.selectbox("Metrik KNN", ["cosine", "euclidean"])
use_numeric = st.sidebar.checkbox("Pakai fitur nutrisi", config.USE_NUMERIC)
num_weight = st.sidebar.slider("Bobot fitur nutrisi", 0.0, 1.0, float(config.NUM_WEIGHT), 0.05,
                               disabled=not use_numeric)
n = st.sidebar.slider("Jumlah rekomendasi (Top-N)", 1, 10, config.TOP_N)

# ---------- Filter ----------
st.sidebar.header("Filter")
veg = st.sidebar.checkbox("Vegetarian saja")
kal_max = int(df["kalori"].max()) if "kalori" in df else 1000
max_kal = st.sidebar.slider("Maks. kalori", 50, kal_max, kal_max, 10)
max_harga = None
if "harga" in df.columns:
    h_max = int(df["harga"].max())
    max_harga = st.sidebar.slider("Maks. harga (Rp)", 1000, h_max, h_max, 1000)
semua_alergi = sorted({a.strip() for x in df["alergi"] for a in str(x).split(";")
                       if a.strip() not in ("", "-")})
alergi = st.sidebar.multiselect("Hindari alergen", semua_alergi)

filters = dict(vegetarian_only=veg, max_kalori=max_kal, max_harga=max_harga, alergi=alergi)


@st.cache_resource
def get_model(_df, key, method, use_numeric, num_weight, knn_metric):
    return FoodRecommender(_df, method=method, use_numeric=use_numeric,
                           num_weight=num_weight, knn_metric=knn_metric)


model = get_model(df, (len(df), tuple(df.columns)), method, use_numeric, num_weight, knn_metric)


def rupiah(x):
    return "Rp" + f"{int(x):,}".replace(",", ".")


def show(recs):
    if recs.empty:
        st.warning("Tidak ada rekomendasi yang cocok dengan filter.")
        return
    for _, r in recs.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.subheader(r["nama"])
            info = f"**{r['kategori']}** • {r['cita_rasa']}"
            if "kalori" in r:
                info += f" • {int(r['kalori'])} kkal"
            if "harga" in r:
                info += f" • {rupiah(r['harga'])}"
            c1.write(info)
            c1.caption(r["alasan"])
            c2.metric("Skor kemiripan", f"{r['skor']:.2f}")
            c2.progress(min(max(r["skor"], 0.0), 1.0))


tab1, tab2, tab3, tab4 = st.tabs(["🍽️ Dari makanan favorit", "🔎 Dari preferensi", "📊 Evaluasi", "📋 Dataset"])

with tab1:
    pilih = st.selectbox("Pilih makanan yang kamu suka", sorted(df["nama"]))
    if st.button("Rekomendasikan", key="b1"):
        show(model.recommend(pilih, n=n, **filters))

with tab2:
    q = st.text_input("Ketik bahan/rasa, contoh: pedas ayam santan")
    if st.button("Cari", key="b2") and q.strip():
        show(model.recommend_query(q, n=n, **filters))

with tab3:
    k = st.select_slider("K untuk Precision@K", config.EVAL_K, value=5 if 5 in config.EVAL_K else config.EVAL_K[0])
    if st.button("Bandingkan model"):
        res = compare_models(df, k)
        st.dataframe(res, use_container_width=True)
        st.bar_chart(res)
    if st.button("Eksperimen bobot nutrisi"):
        res = sweep_weight(df, k)
        st.dataframe(res, use_container_width=True)
        st.line_chart(res[[f"Precision@{k}", "Diversity"]])
    st.caption("Relevan = kategori sama dengan makanan query (leave-one-out, kategori tidak dipakai sebagai fitur).")

with tab4:
    st.dataframe(df.drop(columns="soup", errors="ignore"), use_container_width=True)
    st.caption(f"{len(df)} menu")

