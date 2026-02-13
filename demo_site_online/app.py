
from pathlib import Path



import os, json
import numpy as np
import streamlit as st
import tensorflow as tf

from pipeline import wav_to_model_input


BASE = Path(__file__).parent
MODEL_PATH = str(BASE / "lung_cnn_model.keras")
CLASSES_PATH = str(BASE / "classes.json")
SAMPLES_DIR = str(BASE / "samples")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def list_samples(samples_dir):
    items = []
    if not os.path.isdir(samples_dir):
        return items

    for label in os.listdir(samples_dir):
        label_dir = os.path.join(samples_dir, label)
        if not os.path.isdir(label_dir):
            continue
        if label.startswith(".") or label.startswith("__"):
            continue
        for f in os.listdir(label_dir):
            if f.lower().endswith(".wav"):
                fp = os.path.join(label_dir, f)
                items.append({
                    "filepath": fp,
                    "true_diag": label,
                    "display": f"[{label}] {f}"
                })

    items.sort(key=lambda x: x["display"].lower())
    return items

st.set_page_config(page_title="Predict Sunete (Online Demo)", layout="centered")
st.title("Predict din sunete (DEMO online)")

model = load_model()
classes = load_classes()

items = list_samples(SAMPLES_DIR)
if not items:
    st.error("Nu există samples/. Pune fișiere .wav în samples/<clasă>/ în repo.")
    st.stop()

st.write(f"Fișiere demo disponibile: **{len(items)}**")

choice = st.selectbox("Alege fișier demo:", [it["display"] for it in items])
chosen = next(it for it in items if it["display"] == choice)

audio_path = chosen["filepath"]
true_diag = chosen["true_diag"]

st.audio(audio_path)
st.write(f"**Etichetă reală:** {true_diag}")

if st.button("Fă predicția"):
    x = wav_to_model_input(audio_path)
    prob = model.predict(x, verbose=0)[0]

    pred_idx = int(np.argmax(prob))
    pred_prob = float(prob[pred_idx])
    pred_name = classes[pred_idx] if classes else str(pred_idx)

    st.subheader("Rezultat")
    st.write(f"**Predicție:** {pred_name}")
    st.write(f"**Încredere:** {pred_prob:.3f}")
    st.write(f"**Corect?** {'✅ DA' if pred_name == true_diag else '❌ NU'}")

    st.write("Top 5:")
    topk = np.argsort(prob)[::-1][:5]
    for i in topk:
        name = classes[i] if classes else f"clasa_{i}"
        st.write(f"- {name}: {float(prob[i]):.3f}")