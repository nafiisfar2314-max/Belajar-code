# Belajar-code — Data Science Portfolio (DQLab/BNSP)

Proyek akhir Sertifikasi Data Scientist DQLab. Aplikasi Streamlit
multi-halaman berisi model:

1. **Heart Disease Prediction** — prediksi risiko penyakit jantung
   berdasarkan 9 fitur klinis (model Random Forest, tuned via
   GridSearchCV 5-fold CV, ROC-AUC 0.91, dataset DQLab).

**Demo:** https://belajar-code-bh99xnpsruubztbtjxqsgq.streamlit.app/

## Struktur
- `app.py` — entry point aplikasi Streamlit
- `Model/` — model tersimpan (.pkl)
- `requirements.txt` — dependencies
- `Procfile`, `runtime.txt`, `setup.sh` — konfigurasi deployment

## Cara jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```
