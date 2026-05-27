# Deteksi Lowongan Kerja Palsu dengan BERT, ALBERT, dan RoBERTa

[EN](README.md) | [ID](README.id.md)

Proyek machine learning untuk mendeteksi lowongan kerja palsu dengan klasifikasi teks berbasis Transformer. Fokus riset pada repository ini adalah membandingkan tiga model pretrained Transformer, yaitu BERT, ALBERT, dan RoBERTa. Repository ini juga menyediakan aplikasi web Streamlit untuk inferensi real-time menggunakan model Transformer yang sudah diekspor.

## Hasil

Notebook terbaru yang sudah dieksekusi adalah `research_pipeline.ipynb`, termasuk output run Colab yang sudah selesai. README ini merangkum eksperimen Transformer saja karena pembahasan skripsi berfokus pada BERT, ALBERT, dan RoBERTa.

Dataset: [EMSCAD](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction), berisi 17.880 postingan lowongan kerja dengan 866 postingan fraudulent (fraud rate 4,84%).

### Kesesuaian dengan Bab IV Hasil dan Pembahasan

README ini diselaraskan dengan Bab IV Hasil dan Pembahasan pada skripsi. Alur yang didokumentasikan mengikuti urutan pembahasan pada bab tersebut, mulai dari deskripsi dataset EMSCAD, pembagian data secara stratified, penanganan data tidak seimbang, tokenisasi Transformer, fine-tuning model, evaluasi dengan tuned threshold, interpretasi confusion matrix, sampai deployment model terpilih.

Karena itu, ringkasan eksperimen hanya difokuskan pada BERT, ALBERT, dan RoBERTa. Metrik yang ditampilkan juga mengikuti Bab IV: accuracy digunakan untuk melihat ketepatan umum, precision dan recall menjelaskan trade-off pada kelas fraud, Fraud F1 merangkum keseimbangan keduanya, sedangkan ROC-AUC dan PR-AUC dipakai untuk menilai kualitas pemeringkatan pada dataset yang tidak seimbang. Bagian confusion matrix mengikuti pembahasan Bab IV dengan menunjukkan bagaimana masing-masing model Transformer menghasilkan true positive, true negative, false positive, dan false negative.

### Pembagian Data

Setiap eksperimen memakai split stratified 70/15/15 berdasarkan label `fraudulent`. Run final memakai tiga seed: `42`, `123`, dan `2024`.

| Split | Sampel | Sampel Fraud | Persentase |
|-------|--------|--------------|------------|
| Train | 12.516 | 606 | 70% |
| Validation | 2.682 | 130 | 15% |
| Test | 2.682 | 130 | 15% |

### Konfigurasi Eksperimen

- **Run mode:** `full_multi_seed`
- **Model:** BERT, ALBERT, RoBERTa
- **Seed:** `42`, `123`, `2024`
- **Panjang sequence maksimum:** 256 token
- **Batch size:** 16
- **Epoch:** 5
- **Optimizer:** AdamW, learning rate `2e-5`, weight decay `0.01`, warmup ratio `0.1`
- **Penanganan imbalance:** weighted cross-entropy pada proses fine-tuning Transformer
- **Thresholding:** threshold dituning di validation set, lalu dievaluasi di test set
- **Hardware:** Google Colab Tesla T4 GPU

### Rata-rata Metrik Test dari 3 Seed

Metrik berikut adalah mean +/- standard deviation dengan tuned threshold. Kelas positif pada evaluasi adalah `fraudulent`.

| Model | Accuracy | Precision | Recall | Fraud F1 | ROC-AUC | PR-AUC | Runtime |
|-------|----------|-----------|--------|----------|---------|--------|---------|
| BERT | **0.9886 +/- 0.0024** | **0.9331 +/- 0.0225** | 0.8231 +/- 0.0353 | **0.8745 +/- 0.0273** | **0.9895 +/- 0.0030** | **0.9232 +/- 0.0214** | 28.24 +/- 0.93 menit |
| RoBERTa | 0.9858 +/- 0.0036 | 0.8652 +/- 0.0394 | **0.8385 +/- 0.0407** | 0.8515 +/- 0.0380 | 0.9872 +/- 0.0070 | 0.9106 +/- 0.0247 | **26.13 +/- 3.65 menit** |
| ALBERT | 0.9853 +/- 0.0035 | 0.8875 +/- 0.0292 | 0.8000 +/- 0.0846 | 0.8395 +/- 0.0455 | 0.9849 +/- 0.0131 | 0.9007 +/- 0.0239 | 32.63 +/- 0.14 menit |

**Model Transformer terbaik:** BERT, berdasarkan Fraud F1, precision, ROC-AUC, dan PR-AUC paling kuat pada evaluasi tiga seed.

RoBERTa memiliki recall fraud tertinggi, sehingga model ini menangkap sedikit lebih banyak lowongan palsu. BERT dipilih untuk deployment karena memberi keseimbangan terbaik antara mendeteksi fraud dan menekan peringatan palsu.

### Ringkasan Confusion Matrix

Visualisasi confusion matrix berikut menggabungkan prediksi test set dari tiga seed setelah tuned threshold diterapkan, dengan total 8.046 sampel uji.

![Confusion matrix Transformer](artifacts/figures/transformer_confusion_matrices.png)

| Model | TN | FP | FN | TP | Interpretasi Utama |
|-------|----|----|----|----|--------------------|
| BERT | 7.633 | 23 | 69 | 321 | False positive paling rendah dan precision paling kuat |
| ALBERT | 7.616 | 40 | 78 | 312 | Kasus fraud yang terlewat paling banyak di antara tiga model |
| RoBERTa | 7.605 | 51 | 63 | 327 | Recall fraud terbaik, tetapi false positive lebih tinggi |

### Model Deployment yang Diekspor

Model deployment yang diekspor dari notebook adalah BERT seed `2024`, dengan selected threshold `0.10`.

| Model Export | Accuracy | Precision | Recall | Fraud F1 | PR-AUC | ROC-AUC |
|--------------|----------|-----------|--------|----------|--------|---------|
| BERT seed 2024 | 0.9899 | 0.9328 | 0.8538 | 0.8916 | 0.9239 | 0.9902 |

### Catatan Review Saat Ini

Catatan ini disimpan untuk menjaga reproducibility dan perlu diperiksa sebelum hasil notebook dianggap sebagai eksperimen final yang terkunci.

- **Early stopping dikonfigurasi tetapi tidak aktif pada run Colab.** Output berisi warning berulang bahwa `eval_validation_fraud_f1` tidak ditemukan, sehingga run Transformer sebaiknya dibaca sebagai run 5 epoch sampai hal ini diperbaiki dan dijalankan ulang.
- **Ada warning saat reload checkpoint Transformer.** Run memakai `transformers 5.0.0` dan mengeluarkan warning missing/unexpected LayerNorm keys ketika checkpoint dimuat. Hal ini perlu diinvestigasi sebelum hasil checkpoint dipakai sebagai bukti final paper.

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
artifacts/figures/            # Gambar hasil riset, termasuk visualisasi confusion matrix
research_pipeline.ipynb       # Notebook training dan evaluasi Transformer yang sudah dieksekusi
requirements.txt              # Dependency Python
test_data.txt                 # Sampel untuk testing manual
best_model/                   # Artifact Transformer hasil export, git-ignored
docs/                         # Catatan desain dan dokumen perencanaan
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

Buka `research_pipeline.ipynb` di Jupyter atau Google Colab. Download dataset EMSCAD sebagai `fake_job_postings.csv` dan letakkan di root proyek atau upload di Colab. Jalankan semua cell untuk membuat ulang output eksperimen Transformer dan mengekspor `best_model/`.

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
- **Penanganan class imbalance:** weighted loss pada proses fine-tuning Transformer
- **Perbandingan riset:** BERT, ALBERT, dan RoBERTa pada tiga seed

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

- **Model:** BERT, ALBERT, RoBERTa
- **Training:** PyTorch, Hugging Face Transformers, Hugging Face Trainer
- **Data dan metrik:** pandas, NumPy, scikit-learn
- **Aplikasi:** Streamlit
- **OCR:** pytesseract dengan Tesseract
- **Metrik:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, runtime, inference latency
