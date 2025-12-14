import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Forest Health Classifier (ANN)",
    page_icon="🌳",
    layout="centered"
)

# ----------------------------
# Paths (portable)
# webapp/
#   app.py
#   artifacts/
#   sample_input/
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
SAMPLE_CSV_PATH = os.path.join(BASE_DIR, "sample_input", "example_input.csv")

MODEL_PATH = os.path.join(ARTIFACT_DIR, "deploy_ann_forest.h5")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
CLASSES_PATH = os.path.join(ARTIFACT_DIR, "classes.json")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "feature_names.json")


# ----------------------------
# Load artifacts (cached)
# ----------------------------
@st.cache_resource
def load_artifacts():
    missing = [p for p in [MODEL_PATH, SCALER_PATH, CLASSES_PATH, FEATURES_PATH] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Artifacts tidak lengkap. Pastikan file berikut ada:\n- " + "\n- ".join(missing)
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    # Validasi dasar
    if not isinstance(feature_names, list) or len(feature_names) != 18:
        raise ValueError("feature_names.json harus berupa list berisi 18 nama fitur.")
    if not isinstance(classes, list) or len(classes) != 4:
        raise ValueError("classes.json harus berupa list berisi 4 nama kelas.")

    return model, scaler, classes, feature_names


def validate_input_df(df_in: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    - Pastikan tepat 1 baris
    - Pastikan semua kolom wajib ada
    - Abaikan kolom ekstra
    - Susun ulang kolom sesuai feature_names
    """
    missing_cols = [c for c in feature_names if c not in df_in.columns]
    extra_cols = [c for c in df_in.columns if c not in feature_names]

    if missing_cols:
        raise ValueError(f"Kolom wajib berikut belum ada: {missing_cols}")

    if len(df_in) != 1:
        raise ValueError("CSV harus berisi tepat 1 baris data (1 sampel).")

    # Ambil hanya kolom yang diperlukan & urutkan
    df_fixed = df_in[feature_names].copy()

    # Pastikan numeric
    try:
        df_fixed = df_fixed.astype(float)
    except Exception:
        raise ValueError("Pastikan semua nilai fitur bertipe numerik (angka).")

    return df_fixed, missing_cols, extra_cols


def predict_one(df_fixed: pd.DataFrame, model, scaler, classes: list[str]):
    X = df_fixed.values  # shape (1, 18)
    X_scaled = scaler.transform(X)
    probs = model.predict(X_scaled, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    return pred_label, probs


# ----------------------------
# UI Header
# ----------------------------
st.title("🌳 Forest Health Classifier (ANN)")
st.write(
    "Aplikasi ini memprediksi **kondisi kesehatan pohon** berdasarkan 18 fitur numerik.\n\n"
    "**Cara pakai:** upload file CSV berisi **1 baris** dengan header 18 fitur (tanpa `Plot_ID` dan tanpa `Health_Status`)."
)

with st.expander("📌 Format input yang benar", expanded=False):
    st.markdown(
        "- File **CSV**\n"
        "- **1 baris data** (1 sampel)\n"
        "- Memiliki header kolom sesuai `feature_names.json`\n"
        "- Tidak menyertakan `Plot_ID` dan `Health_Status`\n"
    )

# ----------------------------
# Load artifacts + show status
# ----------------------------
try:
    model, scaler, classes, feature_names = load_artifacts()
    st.success("✅ Artifacts berhasil dimuat (model, scaler, classes, feature names).")
except Exception as e:
    st.error(f"❌ Gagal memuat artifacts: {e}")
    st.stop()

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("⚙️ Opsi")
use_sample = st.sidebar.checkbox("Gunakan contoh dari sample_input (example_input.csv)", value=False)
show_probs = st.sidebar.checkbox("Tampilkan probabilitas semua kelas", value=True)
download_template = st.sidebar.checkbox("Tampilkan tombol download template CSV", value=True)

# Optional: show where sample is expected
st.sidebar.caption(f"Sample path (lokal): {SAMPLE_CSV_PATH}")

# ----------------------------
# Template download
# ----------------------------
if download_template:
    template_df = pd.DataFrame([{col: 0.0 for col in feature_names}])
    st.download_button(
        label="⬇️ Download Template CSV (1 baris)",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="forest_input_template.csv",
        mime="text/csv",
        help="Template berisi header 18 fitur. Isi 1 baris nilai numerik lalu upload kembali."
    )

st.divider()

# ----------------------------
# Input section
# ----------------------------
st.subheader("📤 Input Data")

uploaded = None
df_in = None

col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader("Upload CSV (1 baris)", type=["csv"])

with col2:
    if st.button("📄 Muat contoh example_input.csv", disabled=not use_sample):
        if not os.path.exists(SAMPLE_CSV_PATH):
            st.error("File contoh tidak ditemukan di sample_input/example_input.csv")
        else:
            df_in = pd.read_csv(SAMPLE_CSV_PATH)
            st.info("Contoh berhasil dimuat dari sample_input.")

# If user uploaded, prioritize upload
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        st.stop()

if df_in is None:
    st.warning("Silakan upload CSV, atau centang opsi sample lalu klik tombol muat contoh.")
    st.stop()

# Preview
st.markdown("**Preview input:**")
st.dataframe(df_in, use_container_width=True)

# ----------------------------
# Validate & Predict
# ----------------------------
st.subheader("🧠 Prediksi")

try:
    df_fixed, _, extra_cols = validate_input_df(df_in, feature_names)

    if extra_cols:
        st.warning(f"Kolom ekstra diabaikan: {extra_cols}")

    with st.expander("🔎 Data yang dipakai model (sudah diurutkan)", expanded=False):
        st.dataframe(df_fixed, use_container_width=True)

    if st.button("🚀 Predict Health Status"):
        pred_label, probs = predict_one(df_fixed, model, scaler, classes)

        st.success(f"✅ Prediksi Health_Status: **{pred_label}**")

        if show_probs:
            prob_df = pd.DataFrame({"Class": classes, "Probability": probs})
            prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)
            st.markdown("**Probabilitas per kelas:**")
            st.dataframe(prob_df, use_container_width=True)

            # Quick insight
            top2 = prob_df.head(2)
            if len(top2) == 2:
                gap = float(top2.loc[0, "Probability"] - top2.loc[1, "Probability"])
                st.caption(f"Selisih probabilitas Top-1 vs Top-2: {gap:.3f} (semakin besar biasanya semakin yakin).")

except Exception as e:
    st.error(f"❌ Input tidak valid: {e}")
    st.stop()

# Footer
st.divider()
st.caption("Catatan: hasil prediksi bergantung pada kualitas data input dan distribusi kelas pada dataset training.")
