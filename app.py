import gradio as gr
import skops.io as sio

# Load model
pipe = sio.load("./Model/credit_loan_detection.skops", trusted=True)

# Fungsi prediksi
def predict_credit(age, income, job_type, credit_score, loan_amount):
    """Predict credit approval based on applicant features."""
    features = [age, income, job_type, credit_score, loan_amount]
    predicted_label = pipe.predict([features])[0]
    label = f"Hasil Prediksi: {predicted_label}"
    return label

# Input untuk Gradio
inputs = [
    gr.Slider(18, 70, step=1, label="Usia Pemohon (tahun)"),
    gr.Slider(0, 100_000_000, step=500_000, label="Penghasilan Bulanan (Rp)"),
    gr.Radio(["PNS", "Swasta", "Wiraswasta", "Pelajar", "Lainnya"], label="Jenis Pekerjaan"),
    gr.Slider(300, 850, step=1, label="Skor Kredit"),
    gr.Slider(1_000_000, 500_000_000, step=1_000_000, label="Jumlah Pinjaman (Rp)"),
]

# Output label
outputs = gr.Label(num_top_classes=3)

# Contoh input untuk testing
examples = [
    [30, 5_000_000, "Swasta", 700, 50_000_000],
    [45, 12_000_000, "PNS", 760, 150_000_000],
    [22, 2_500_000, "Pelajar", 600, 10_000_000],
]

# UI metadata
title = "Klasifikasi Kelayakan Kredit"
description = "Masukkan data calon peminjam untuk mengetahui hasil prediksi kelayakan kredit menggunakan model machine learning."
article = "Aplikasi ini menggunakan model SKOPS dan Gradio untuk memprediksi kelayakan kredit berdasarkan fitur data calon peminjam."

# Interface
gr.Interface(
    fn=predict_credit,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft()
).launch()
