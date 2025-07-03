import os
import re
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# Tentukan path model dan tokenizer
MODEL_PATH = 'sentiment_lstm_model.h5'
TOKENIZER_PATH = 'tokenizer.pkl'
MAX_LEN = 100  # Harus sama dengan MAX_LEN saat pelatihan model

# Muat model dan tokenizer saat aplikasi dimulai
try:
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Model dan tokenizer berhasil dimuat!")
except Exception as e:
    print(f"Error memuat model atau tokenizer: {e}")

# Definisikan pemetaan label
label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'} #Harus sesuai dengan to_categorical saat pelatihan

# Fungsi untuk membersihkan teks (harus sama dengan yang digunakan saat pelatihan)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    return text

@app.route('/')
def home():
    
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json(force=True)
            text_to_predict = data['text']

            # Bersihkan teks
            cleaned_text = clean_text(text_to_predict)

            # Tokenisasi dan padding
            sequence = tokenizer.texts_to_sequences([cleaned_text])
            padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')

            # Prediksi
            prediction = model.predict(padded_sequence)
            predicted_class = np.argmax(prediction, axis=1)[0]
            sentiment = label_mapping.get(predicted_class, 'unknown') # Mendapatkan nama sentimen

            # Mendapatkan probabilitas untuk setiap kelas
            probabilities = prediction[0].tolist()
            sentiment_scores = {label_mapping[i]: prob for i, prob in enumerate(probabilities)}

            response = {
                'original_text': text_to_predict,
                'cleaned_text': cleaned_text,
                'predicted_sentiment': sentiment,
                'confidence_scores': sentiment_scores
            }
            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Pastikan model dan tokenizer ada sebelum menjalankan aplikasi
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Error: Tokenizer file not found at {TOKENIZER_PATH}")
        exit()

    app.run(debug=True) # debug=True akan me-reload server secara otomatis saat ada perubahan kode