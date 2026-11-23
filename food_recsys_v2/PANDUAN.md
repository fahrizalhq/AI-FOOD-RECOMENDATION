# Panduan Lengkap: AI Food Recommendation System

Bahasa: **Python** saja. Semua pengaturan ada di `config.py`.

## Struktur folder

```
Food_recomendation/
├── data/
│   └── makanan.csv        <- dataset (56 menu, 11 kolom)
├── config.py              <- PUSAT PENGATURAN (ubah di sini)
├── recommender.py         <- model (TF-IDF, Cosine, KNN, filter)
├── evaluate.py            <- evaluasi dan eksperimen
├── app.py                 <- aplikasi web Streamlit
└── requirements.txt       <- daftar library
```

---

## TAHAP 1 — Persiapan

1. Ekstrak zip, salin SEMUA isinya ke folder proyek PyCharm.
   `data/makanan.csv` harus tetap di dalam folder `data`.
2. Buka terminal PyCharm di folder proyek, lalu instal library:
   ```
   pip install -r requirements.txt
   ```
   (Peringatan "not on PATH" boleh diabaikan.)
3. Cek berhasil: `python -c "import pandas, sklearn, streamlit; print('OK')"`

## TAHAP 2 — Cek data

Kolom `makanan.csv`:

| Kolom | Keterangan | Wajib |
|---|---|---|
| nama | nama makanan (unik) | Ya |
| kategori | Nasi, Mie, Sup, Sayur, Ayam, Daging, Sate, Gorengan, Dessert, Seafood | Ya |
| bahan | dipisah koma | Ya |
| cita_rasa | contoh: gurih pedas | Ya |
| kalori, protein, lemak, karbohidrat | angka (perkiraan) | Tidak |
| harga | rupiah, untuk filter anggaran | Tidak |
| vegetarian | ya / tidak | Tidak |
| alergi | pisahkan dengan `;`, tulis `-` jika tidak ada | Tidak |

Catatan laporan: nilai gizi dan harga pada data contoh adalah **perkiraan**.
Tulis ini sebagai asumsi/keterbatasan, atau ganti dengan data resmi (TKPI Kemenkes).

Cara menambah menu: tambah baris baru di akhir `makanan.csv` dengan format yang sama.
Kalau mau memperbanyak data, tambah sampai 150-300 menu.

## TAHAP 3 — Jalankan model (tanpa web)

```
python -c "from recommender import *; m=FoodRecommender(load_data()); print(m.recommend('Nasi Goreng', 5))"
```

Kalau muncul 5 rekomendasi, model sudah jalan.

## TAHAP 4 — Evaluasi

```
python evaluate.py
```

Menghasilkan: perbandingan 5 konfigurasi model (K = 3, 5, 10) dan eksperimen
bobot nutrisi (disimpan ke `hasil_eksperimen_bobot.csv`).

Arti metrik:
- **Precision@K**: porsi rekomendasi yang kategorinya sama dengan makanan query.
- **Coverage**: porsi menu yang pernah muncul sebagai rekomendasi.
- **Diversity**: keragaman isi daftar rekomendasi (makin tinggi makin beragam).

## TAHAP 5 — Jalankan web

```
python -m streamlit run app.py
```

Buka `http://localhost:8501`. Fitur: pilih makanan favorit, ketik preferensi,
filter vegetarian/kalori/harga/alergi, ganti algoritma, geser bobot nutrisi,
tab evaluasi dan tab dataset. Berhenti dengan Ctrl+C di terminal.

---

## PARAMETER YANG BISA KAMU UBAH (config.py)

| Parameter | Nilai awal | Coba | Efek |
|---|---|---|---|
| METHOD | "cosine" | "knn" | algoritma pencari kemiripan |
| KNN_METRIC | "cosine" | "euclidean" | jarak untuk KNN (mengikuti jurnal Almas dkk.) |
| USE_NUMERIC | True | False | pakai/tidak fitur nutrisi |
| NUM_WEIGHT | 0.5 | 0 - 1 | pengaruh nutrisi terhadap hasil |
| NUM_COLS | kalori, protein, lemak, karbohidrat | tambah "harga" | kolom angka yang dihitung |
| TEXT_COLS | bahan, cita_rasa, kategori | hapus "kategori" | kolom teks untuk TF-IDF |
| TOP_N | 5 | 3 - 10 | jumlah rekomendasi |
| EVAL_K | [3, 5, 10] | [5] | K untuk Precision@K |
| WEIGHT_SWEEP | 0, .25, .5, .75, 1 | tambah nilai | bobot yang diuji di eksperimen |

Cara bereksperimen: ubah SATU parameter, jalankan `python evaluate.py`,
catat hasilnya.

### Template tabel hasil untuk laporan

| Percobaan | Algoritma | Metrik | Bobot nutrisi | Precision@5 | Coverage | Diversity |
|---|---|---|---|---|---|---|
| 1 | Cosine | - | 0 | | | |
| 2 | Cosine | - | 0.5 | | | |
| 3 | KNN | cosine | 0.5 | | | |
| 4 | KNN | euclidean | 0.5 | | | |

---

## TAHAP 6 — Analisis dan perbaikan sistem (tugas 9)

Hal yang ditemukan dari hasil awal (data contoh 56 menu), bisa kamu cek ulang:
- Cosine dan KNN (metrik cosine) memberi hasil identik, karena secara matematis sama.
  KNN dengan **euclidean** memberi hasil berbeda, jadi cocok untuk perbandingan.
- Menaikkan bobot nutrisi menaikkan Precision tetapi menurunkan Diversity (trade-off).
- Precision tampak rendah karena relevansi diukur hanya dari kategori yang sama.

Ide perbaikan berurutan dari yang termudah:
1. Atur bobot nutrisi ke titik seimbang (misal 0.5) dan jelaskan alasannya.
2. Tambah data menjadi 150+ menu.
3. Tambah "harga" ke `NUM_COLS` lalu lihat efeknya.
4. Stemming bahasa Indonesia: `pip install PySastrawi`, lalu di `preprocess()`
   (recommender.py) tambahkan:
   ```python
   from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
   stemmer = StemmerFactory().create_stemmer()
   # di akhir preprocess(): return stemmer.stem(hasil)
   ```
5. Bigram TF-IDF: ubah `TfidfVectorizer()` menjadi `TfidfVectorizer(ngram_range=(1, 2))`.
6. User study kecil: 10-20 responden menilai kecocokan hasil (skala 1-5).

## Rujukan dalam laporan

- Almas, Susanti, & Handajani (2024). *Statistika*, 24(1), 115-122.
- Azhar dkk. (2026). *JTIK*, 10(1), 13-21.
- Nafi & Mujiyono (2025). *Jurnal Algoritma*, 22(2), 381-388.

Posisi penelitianmu: gabungan kemiripan teks (bahan dan cita rasa, TF-IDF)
dengan fitur nutrisi, dibandingkan antara Cosine dan KNN.

## Masalah umum

| Pesan error | Penyebab dan solusi |
|---|---|
| ModuleNotFoundError | library belum terinstal: jalankan `pip install -r requirements.txt` |
| FileNotFoundError data/makanan.csv | terminal tidak di folder proyek, atau folder `data` hilang |
| Kolom wajib tidak ada | nama kolom di CSV tidak sesuai tabel Tahap 2 |
| 'streamlit' tidak dikenali | pakai `python -m streamlit run app.py` |
| Rekomendasi kosong | filter terlalu ketat (alergi/harga/kalori), longgarkan filternya |
