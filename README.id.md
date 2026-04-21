# Job Scam Detection menggunakan BERT, ALBERT & RoBERTa

[EN](README.md) | [ID](README.id.md)

Sistem machine learning untuk mendeteksi lowongan kerja palsu menggunakan model NLP berbasis transformer. Terdiri dari research notebook untuk training/perbandingan model dan aplikasi web Streamlit untuk inferensi real-time.

## Hasil

Dilatih pada [dataset EMSCAD](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) — 17,880 postingan lowongan kerja, 4.84% berlabel fraudulent (sangat imbalanced).

### Pembagian Data

Stratified berdasarkan label `fraudulent` (random seed `42`) sehingga rasio fraud (~4.84%) tetap terjaga di semua split.

| Split      | Jumlah Sampel | Persentase |
|------------|---------------|------------|
| Train      | 12,516        | 70%        |
| Validation | 2,682         | 15%        |
| Test       | 2,682         | 15%        |

### Konfigurasi Training

- **Panjang sequence maksimum:** 256 token
- **Batch size:** 16
- **Epoch:** 3
- **Optimizer:** AdamW — learning rate `2e-5`, weight decay `0.01`, warmup ratio `0.1`
- **Loss:** Weighted cross-entropy — class weights `[0.5254, 10.3267]` untuk mengimbangi ketidakseimbangan kelas
- **Pemilihan model:** checkpoint terbaik berdasarkan F1 di validation set (`load_best_model_at_end=True`)
- **Hardware:** single Tesla T4 GPU (Google Colab)

### Metrik pada Test Set

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| BERT | 0.9870 | 0.9130 | 0.8077 | 0.8571 |
| **ALBERT** | **0.9892** | **0.9469** | **0.8231** | **0.8807** |
| RoBERTa | 0.9847 | 0.8678 | 0.8077 | 0.8367 |

**Model terbaik: ALBERT** (albert-base-v2) — dipilih berdasarkan F1 score tertinggi.  
Download best_model: https://drive.google.com/file/d/1YZoeuoWHS_oLGF4bu6cW3ePcbaBGEdB2/view?usp=sharing

## Struktur Proyek

```
├── app.py                       # Entry point (mendelegasikan ke AppController)
├── src/                         # Kode aplikasi MVC
│   ├── controllers/
│   │   └── app_controller.py    # Mengorkestrasikan input → preprocess → predict → render
│   ├── models/
│   │   ├── classifier.py        # Memuat ALBERT, menjalankan inferensi → (label, confidence)
│   │   ├── ocr_engine.py        # Wrapper untuk Tesseract OCR
│   │   └── preprocessor.py      # Pembersihan teks (mirror dari notebook)
│   └── views/
│       └── main_view.py         # Rendering UI Streamlit
├── research_pipeline.ipynb      # Notebook training & perbandingan model
├── requirements.txt             # Dependencies Python
├── test_data.txt                # Dua sampel postingan (legit + scam) untuk testing manual
├── best_model/                  # Model ALBERT hasil export + model_meta.json (di-ignore git)
└── docs/                        # Catatan desain dan dokumen perencanaan
```

## Arsitektur

Aplikasi mengikuti pemisahan Model–View–Controller agar UI Streamlit, logika bisnis, dan urusan ML tetap terpisah.

- **View** (`src/views/main_view.py`) — rendering Streamlit murni (header, form input, tampilan hasil). Stateless; menerima data untuk di-render dan callback untuk dipanggil.
- **Model** (`src/models/`) — logika domain, bukan bobot ML. `classifier.py` memuat ALBERT yang sudah di-fine-tune dan mengembalikan `(label, confidence)`; `preprocessor.py` mereplikasi pembersihan teks dari notebook supaya training dan inferensi tetap selaras; `ocr_engine.py` membungkus Tesseract untuk upload gambar.
- **Controller** (`src/controllers/app_controller.py`) — menghubungkan semuanya: memuat classifier (di-cache lewat `@st.cache_resource`), meneruskan input user melalui preprocessor dan classifier, lalu mengirim hasilnya kembali ke view.

`app.py` hanyalah entry point tipis yang membuat instance `AppController`, jadi `streamlit run app.py` tetap berjalan seperti biasa.

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (untuk fitur upload gambar)

Fitur upload gambar membutuhkan Tesseract OCR untuk mengekstrak teks dari screenshot.

- **Windows**: Download dan install dari [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Path instalasi default (`C:\Program Files\Tesseract-OCR`) akan terdeteksi otomatis oleh aplikasi.
- **Linux**: `sudo apt install tesseract-ocr`
- **Mac**: `brew install tesseract`

> Jika Anda hanya memakai fitur input teks, Tesseract tidak diperlukan.

### 3. Dapatkan model weights

Aplikasi memuat `./best_model/` saat startup dan akan langsung gagal jika folder tersebut tidak ada. Pilih salah satu cara:

**Opsi A — Download ALBERT pre-trained (paling cepat)**

[Download `best_model.zip`](https://drive.google.com/file/d/1YZoeuoWHS_oLGF4bu6cW3ePcbaBGEdB2/view?usp=sharing) lalu ekstrak di root proyek sehingga `./best_model/` berisi model weights, tokenizer, dan `model_meta.json`.

**Opsi B — Training dari awal**

Buka `research_pipeline.ipynb` di Jupyter atau Google Colab. Download [dataset EMSCAD](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) sebagai `fake_job_postings.csv` dan letakkan di root proyek. Jalankan semua cell — model terbaik akan di-export ke `best_model/`.

GPU sangat disarankan (training butuh ~2 jam di Tesla T4).

### 4. Jalankan aplikasi web

```bash
streamlit run app.py
```

Buka http://localhost:8501 di browser Anda. Untuk sanity-check setup, copy salah satu sampel dari `test_data.txt` (satu legitimate, satu scam yang jelas) ke input teks lalu klik **Analyze**.

## Fitur

- **Input teks** — paste deskripsi lowongan langsung
- **Upload gambar** — upload screenshot; teks diekstrak via OCR (pytesseract)
- **Confidence score** — menampilkan probabilitas prediksi
- **Weighted loss** — menangani class imbalance (tingkat fraud 4.84%) dengan weighted cross-entropy

## Keterbatasan & Penggunaan yang Bertanggung Jawab

Anggap output model sebagai **sinyal screening, bukan vonis akhir.** Gabungkan dengan penilaian Anda sendiri dan pengecekan eksternal (legalitas perusahaan, identitas rekruter, umur domain, ada tidaknya permintaan pembayaran di muka, dll.).

- **~18% scam lolos deteksi.** Recall di test set = 0.8231 — sekitar 1 dari 5 postingan fraudulent di test set kami dilabeli "Legitimate Job". Jangan pernah anggap prediksi "Legitimate Job" sebagai jaminan keamanan.
- **~5% flag adalah false alarm.** Precision di test set = 0.9469. Flag "Potential Scam" artinya perlu diperiksa lebih teliti, bukan penolakan otomatis.
- **Hanya bahasa Inggris.** EMSCAD adalah dataset sepenuhnya berbahasa Inggris. Lowongan dalam bahasa Indonesia, Spanyol, atau bahasa lain adalah out of distribution dan hasilnya tidak reliable.
- **Dataset drift.** EMSCAD dikumpulkan pada tahun 2014–2015. Pola scam yang lebih baru (umpan crypto, postingan yang digenerate AI, skema MLM yang lebih canggih) mungkin lolos dari model yang dilatih di data lama.
- **Truncation 256 token.** Postingan yang lebih panjang dari ~256 token akan dipotong — sinyal yang hanya muncul di bagian akhir deskripsi panjang tidak akan terlihat oleh model.
- **Reliabilitas OCR.** Upload gambar bergantung pada Tesseract. Screenshot resolusi rendah, miring, atau noisy menghasilkan ekstraksi teks yang buruk, yang menurunkan kualitas prediksi.
- **Bukan nasihat hukum atau finansial.** Jangan jadikan tool ini sebagai satu-satunya dasar untuk menerima, menolak, atau melaporkan tawaran kerja.

## Tech Stack

- **Model**: BERT, ALBERT, RoBERTa (HuggingFace Transformers)
- **Training**: PyTorch, HuggingFace Trainer dengan custom weighted loss
- **Aplikasi**: Streamlit
- **OCR**: pytesseract (butuh [Tesseract](https://github.com/tesseract-ocr/tesseract) terinstal)
- **Metrik**: scikit-learn (accuracy, precision, recall, F1, confusion matrix)
