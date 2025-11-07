import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- 1. INISIALISASI APLIKASI FLASK ---
app = Flask(__name__)
# CORS(app) mengizinkan aplikasi HP (dari domain berbeda) mengakses server ini
CORS(app) 

# --- 2. KONFIGURASI PATH (OTOMATIS) ---
# Kodingan ini secara otomatis menemukan folder tempat ia dijalankan
# Ini akan bekerja di D:\... Anda dan di server cloud
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = '.' 

MODEL_FILE = os.path.join(BASE_DIR, 'pH_model.pkl')
SCALER_FILE = os.path.join(BASE_DIR, 'pH_scaler.pkl')

# --- 3. MUAT MODEL "OTAK" JST (HANYA SEKALI) ---
try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("--- Model dan Scaler berhasil dimuat. Server siap. ---")
except IOError:
    print(f"--- ERROR: Gagal memuat '{MODEL_FILE}' atau '{SCALER_FILE}'. ---")
    print("--- PASTIKAN Anda sudah menjalankan 'train_pH_model.py' terlebih dahulu. ---")
    model = None
    scaler = None

# --- 4. BUAT "ENDPOINT" API (/prediksi) ---
# Ini adalah "link" yang akan dipanggil oleh HP Anda
@app.route('/prediksi', methods=['POST'])
def predict_ph():
    if model is None or scaler is None:
        return jsonify({'error': 'Model tidak siap. Cek log server.'}), 500

    try:
        # 1. Ambil data JSON yang dikirim oleh HP
        data = request.json
        
        # 2. Ambil 6 nilai RGB
        input_list = [
            data['r_ref'], data['g_ref'], data['b_ref'],
            data['r_sample'], data['g_sample'], data['b_sample']
        ]

        # 3. Ubah jadi format numpy
        input_data = np.array([input_list])
        
        # 4. Gunakan Scaler
        input_scaled = scaler.transform(input_data)
        
        # 5. Lakukan Prediksi
        prediksi = model.predict(input_scaled)
        ph_value = prediksi[0]
        
        # 6. Kirim jawaban kembali ke HP
        return jsonify({
            'prediksi_ph': round(ph_value, 2)
        })

    except KeyError:
        return jsonify({'error': 'Data JSON tidak lengkap. Butuh 6 kunci RGB.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 5. BUAT ENDPOINT "HOMEPAGE" (AGAR TIDAK "NOT FOUND") ---
@app.route('/', methods=['GET'])
def home():
    # Halaman ini hanya untuk mengecek apakah server hidup
    return "Selamat! Server JST (ANN) Anda sudah online. Gunakan endpoint /prediksi untuk menebak pH."

# --- 6. JALANKAN SERVER ---
if __name__ == '__main__':
    # Menjalankan server di jaringan lokal Anda (port 5000)
    print("--- Menjalankan server Flask di http://127.0.0.1:5000 ---")
    app.run(host='0.0.0.0', port=5000, debug=True)