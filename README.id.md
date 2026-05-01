# Deteksi Lowongan Kerja Palsu dengan Transformer dan Baseline TF-IDF

[EN](README.md) | [ID](README.id.md)

Proyek machine learning untuk mendeteksi lowongan kerja palsu menggunakan baseline NLP klasik dan model berbasis Transformer. Repository ini berisi research notebook untuk training, perbandingan model, export artifact, dan aplikasi web Streamlit untuk inferensi real-time.

## Hasil

Notebook terbaru yang sudah dieksekusi adalah `research_pipeline.ipynb`, termasuk output run Colab yang sudah selesai.

Dataset: [EMSCAD](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction), berisi 17.880 postingan lowongan kerja dengan 866 postingan fraudulent (fraud rate 4,84%).

### Pembagian Data

Setiap eksperimen memakai split stratified 70/15/15 berdasarkan label `fraudulent`. Run final memakai tiga seed: `42`, `123`, dan `2024`.

| Split | Sampel | Sampel Fraud | Persentase |
|-------|--------|--------------|------------|
| Train | 12.516 | 606 | 70% |
| Validation | 2.682 | 130 | 15% |
| Test | 2.682 | 130 | 15% |

### Konfigurasi Eksperimen

- **Run mode:** `full_multi_seed`
- **Model:** TF-IDF + Logistic Regression, TF-IDF + Linear SVM, BERT, ALBERT, RoBERTa
- **Seed:** `42`, `123`, `2024`
- **Panjang sequence maksimum:** 256 token
- **Batch size:** 16
- **Epoch:** 5
- **Optimizer untuk Transformer:** AdamW, learning rate `2e-5`, weight decay `0.01`, warmup ratio `0.1`
- **Penanganan imbalance:** weighted cross-entropy untuk Transformer; class-balanced classical baseline
- **Thresholding:** threshold dituning di validation set, lalu dievaluasi di test set
- **Hardware:** Google Colab Tesla T4 GPU

### Rata-rata Metrik Test dari 3 Seed

Metrik berikut adalah mean +/- standard deviation dengan tuned threshold.

| Model | Grup | Accuracy | Precision | Recall | Fraud F1 | PR-AUC | Runtime |
|-------|------|----------|-----------|--------|----------|--------|---------|
| TF-IDF + Linear SVM | Klasik | 0.9906 +/- 0.0023 | 0.9509 +/- 0.0144 | 0.8487 +/- 0.0364 | **0.8968 +/- 0.0264** | **0.9449 +/- 0.0125** | 0.28 +/- 0.02 menit |
| BERT | Transformer | 0.9886 +/- 0.0024 | 0.9331 +/- 0.0225 | 0.8231 +/- 0.0353 | 0.8745 +/- 0.0273 | 0.9232 +/- 0.0214 | 28.24 +/- 0.93 menit |
| TF-IDF + Logistic Regression | Klasik | 0.9866 +/- 0.0017 | 0.8627 +/- 0.0346 | **0.8615 +/- 0.0204** | 0.8617 +/- 0.0157 | 0.9261 +/- 0.0102 | 0.28 +/- 0.02 menit |
| RoBERTa | Transformer | 0.9858 +/- 0.0036 | 0.8652 +/- 0.0394 | 0.8385 +/- 0.0407 | 0.8515 +/- 0.0380 | 0.9106 +/- 0.0247 | 26.13 +/- 3.65 menit |
| ALBERT | Transformer | 0.9853 +/- 0.0035 | 0.8875 +/- 0.0292 | 0.8000 +/- 0.0846 | 0.8395 +/- 0.0455 | 0.9007 +/- 0.0239 | 32.63 +/- 0.14 menit |

**Model terbaik overall:** TF-IDF + Linear SVM, berdasarkan mean tuned fraud F1.

**Model deployment yang diekspor dari notebook:** BERT, seed `2024`, selected threshold `0.10`. Notebook saat ini hanya mengekspor model Transformer untuk alur aplikasi, sehingga BERT dipilih sebagai Transformer terbaik untuk deployment walaupun Linear SVM adalah model terbaik overall.

| Model Export | Accuracy | Precision | Recall | Fraud F1 | PR-AUC | ROC-AUC |
|--------------|----------|-----------|--------|----------|--------|---------|
| BERT seed 2024 | 0.9899 | 0.9328 | 0.8538 | 0.8916 | 0.9239 | 0.9902 |

### Catatan Review Saat Ini

Catatan ini sengaja didokumentasikan apa adanya; belum difix di notebook.

- **Early stopping dikonfigurasi tetapi tidak aktif pada run Colab.** Output berisi warning berulang bahwa `eval_validation_fraud_f1` tidak ditemukan, sehingga early stopping dinonaktifkan. Anggap run Transformer sebagai run 5 epoch sampai ini difix dan dirun ulang.
- **Ada warning saat reload checkpoint Transformer.** Run memakai `transformers 5.0.0` dan mengeluarkan warning missing/unexpected LayerNorm keys ketika checkpoint dimuat. Ini perlu diinvestigasi sebelum hasil checkpoint Transformer dipakai sebagai bukti final paper.
- **Narasi paper harus membedakan best overall dan app export.** Linear SVM adalah model terbaik overall pada hasil saat ini; BERT adalah model Transformer yang diekspor untuk workflow aplikasi.

## Struktur Proyek

```text
app.py                       # Entry point Streamlit
src/                         # Kode aplikasi MVC
  controllers/
    app_controller.py         # Mengorkestrasikan input -> preprocess -> predict -> render
  models/
    classifier.py             # Memuat ./best_model dan menjalankan inferensi
    ocr_engine.py             # Wrapper Tesseract OCR
    preprocessor.py           # Pembersihan teks selaras dengan notebook
  views/
    main_view.py              # Rendering UI Streamlit
research_pipeline.ipynb      # Notebook training dan perbandingan model yang sudah dieksekusi
requirements.txt             # Dependency Python
test_data.txt                # Sampel untuk testing manual
best_model/                  # Artifact Transformer hasil export, git-ignored
docs/                        # Catatan desain dan dokumen perencanaan
```

## Arsitektur

Aplikasi mengikuti pemisahan Model-View-Controller agar UI Streamlit, logika bisnis, dan urusan ML tetap terpisah.

- **View** (`src/views/main_view.py`): rendering Streamlit untuk header, form input, dan tampilan hasil.
- **Model** (`src/models/`): logika inferensi dan preprocessing. `classifier.py` memuat model dari `./best_model`, `preprocessor.py` mencerminkan pembersihan teks notebook, dan `ocr_engine.py` membungkus Tesseract untuk upload gambar.
- **Controller** (`src/controllers/app_controller.py`): menghubungkan view, preprocessor, classifier, dan flow OCR.

`app.py` adalah entry point tipis yang membuat `AppController`, sehingga `streamlit run app.py` menjalankan aplikasi.

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Install Tesseract OCR

Fitur upload gambar membutuhkan Tesseract OCR untuk mengekstrak teks dari screenshot.

- **Windows:** install dari [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Path default (`C:\Program Files\Tesseract-OCR`) akan terdeteksi otomatis oleh aplikasi.
- **Linux:** `sudo apt install tesseract-ocr`
- **Mac:** `brew install tesseract`

Jika hanya memakai input teks, Tesseract tidak diperlukan.

### 3. Dapatkan atau train model weights

Aplikasi memuat `./best_model/` saat startup dan akan gagal lebih awal jika folder ini tidak ada.

**Opsi A: Gunakan artifact Transformer export terbaru**

[Download `best_model`](https://drive.google.com/file/d/1r6ikkGpA2ONTEs3A9TzMWvHDIcNazPYF/view?usp=drive_link) lalu ekstrak di root proyek sehingga `./best_model/` berisi model weights, tokenizer, dan `model_meta.json`. Pada run terbaru, model yang diekspor adalah BERT seed `2024`.

**Opsi B: Training dari awal**

Buka `research_pipeline.ipynb` di Jupyter atau Google Colab. Download dataset EMSCAD sebagai `fake_job_postings.csv` dan letakkan di root proyek atau upload di Colab. Jalankan semua cell untuk membuat ulang output eksperimen dan mengekspor `best_model/`.

Run penuh terbaru selesai di Google Colab Free dengan GPU Tesla T4.

### 4. Jalankan aplikasi web

```bash
streamlit run app.py
```

Buka http://localhost:8501 di browser. Untuk sanity-check setup, salin salah satu sampel dari `test_data.txt` ke input teks lalu klik **Analyze**.

## Fitur

- **Input teks:** paste deskripsi lowongan langsung
- **Upload gambar:** upload screenshot; teks diekstrak via OCR
- **Confidence score:** menampilkan confidence prediksi
- **Penanganan class imbalance:** weighted loss dan class-balanced baseline
- **Perbandingan riset:** baseline klasik plus BERT, ALBERT, dan RoBERTa di tiga seed

## Keterbatasan dan Penggunaan Bertanggung Jawab

Anggap output model sebagai sinyal screening, bukan vonis akhir. Gabungkan dengan pengecekan eksternal seperti legalitas perusahaan, identitas rekruter, umur domain, klaim gaji tidak realistis, dan permintaan pembayaran.

- **False negative masih mungkin.** Run BERT yang diekspor memiliki recall `0.8538`, jadi sebagian postingan fraudulent masih bisa lolos.
- **False positive masih mungkin.** Run BERT yang diekspor memiliki precision `0.9328`; flag scam berarti perlu review, bukan penolakan otomatis.
- **Hanya bahasa Inggris.** EMSCAD adalah dataset berbahasa Inggris. Bahasa lain berada di luar distribusi data.
- **Dataset drift.** EMSCAD lebih lama daripada pola scam saat ini, sehingga taktik baru mungkin belum tertangkap.
- **Truncation 256 token.** Sinyal yang muncul di bagian akhir deskripsi panjang bisa terpotong.
- **Reliabilitas OCR.** Upload gambar bergantung pada kualitas ekstraksi Tesseract.
- **Bukan nasihat hukum atau finansial.** Jangan gunakan tool ini sebagai satu-satunya dasar untuk menerima, menolak, atau melaporkan tawaran kerja.

## Tech Stack

- **Model:** TF-IDF + Logistic Regression, TF-IDF + Linear SVM, BERT, ALBERT, RoBERTa
- **Training:** PyTorch, HuggingFace Trainer, scikit-learn
- **Aplikasi:** Streamlit
- **OCR:** pytesseract dengan Tesseract
- **Metrik:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, runtime, inference latency
