# Proyek Analisis Sentimen (Sentiment Analysis Project)

Proyek ini adalah aplikasi web sederhana menggunakan Flask untuk melakukan analisis sentimen terhadap teks masukan. Model analisis sentimen dibangun dengan Keras (TensorFlow) dan menggunakan arsitektur LSTM.

## Fitur

- Analisis sentimen terhadap teks berita yang diberikan (Negative, Neutral, Positive).
- Menampilkan probabilitas kepercayaan untuk setiap kelas sentimen.
- Antarmuka web dasar untuk memasukkan teks.
- API endpoint untuk prediksi sentimen.

## Struktur Proyek

├── sentiment_lstm_model.h5 # Model Keras yang telah dilatih
├── tokenizer.pkl # Objek tokenizer yang telah dilatih
├── app.py  
├── templates/
│ └── index.html # Template HTML
└── README.md

## Instalasi

Ikuti langkah-langkah di bawah ini untuk menginstal dan menjalankan proyek ini.

1. python -m venv venv
2. .\venv\Scripts\activate
3. pip install flask tensorflow keras numpy scikit-learn
4. python app.py

### Catatan:

- Step 1-3 sekali saat pertama kali run program
- Step 4 dijalankan setiap kali run program
