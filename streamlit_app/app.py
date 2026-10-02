from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Natural Scene Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_FILENAME = "final_mobilenetv2_transfer_model.keras"
MODEL_PATH = Path(__file__).resolve().parent / MODEL_FILENAME

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
DISPLAY_NAMES = {
    "buildings": "Buildings",
    "forest": "Forest",
    "glacier": "Glacier",
    "mountain": "Mountain",
    "sea": "Sea",
    "street": "Street",
}
CLASS_ICONS = {
    "buildings": "🏢",
    "forest": "🌲",
    "glacier": "🧊",
    "mountain": "⛰️",
    "sea": "🌊",
    "street": "🛣️",
}
TARGET_SIZE = (224, 224)

st.markdown(
    '''
    <style>
        .block-container {max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem;}
        .hero {padding: 2rem 2.2rem; border-radius: 20px;
               background: linear-gradient(135deg, rgba(79,70,229,.18), rgba(14,165,233,.10));
               border: 1px solid rgba(148,163,184,.25); margin-bottom: 1.5rem;}
        .hero h1 {margin: 0 0 .45rem 0; font-size: 2.45rem;}
        .hero p {margin: 0; font-size: 1.05rem; opacity: .85;}
        .prediction-card {padding: 1.3rem 1.5rem; border-radius: 16px;
                          border: 1px solid rgba(148,163,184,.25);
                          background: rgba(148,163,184,.06); margin-bottom: 1rem;}
        .prediction-label {font-size: .9rem; opacity: .70; margin-bottom: .25rem;}
        .prediction-value {font-size: 2rem; font-weight: 700; margin-bottom: .2rem;}
        .confidence-value {font-size: 1.25rem; font-weight: 650;}
        div[data-testid="stMetric"] {border: 1px solid rgba(148,163,184,.20);
                                   padding: .8rem 1rem; border-radius: 14px;
                                   background: rgba(148,163,184,.04);}
    </style>
    ''',
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def load_scene_model(model_path: str) -> tf.keras.Model:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file was not found at: {path}")

    model = tf.keras.models.load_model(path, compile=False)

    if int(model.output_shape[-1]) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} output classes, found {model.output_shape[-1]}."
        )
    return model

def prepare_image(image: Image.Image):
    rgb_image = image.convert("RGB")
    resized_image = rgb_image.resize(TARGET_SIZE)
    image_array = np.asarray(resized_image, dtype=np.float32)
    input_batch = np.expand_dims(image_array, axis=0)
    return rgb_image, input_batch

def predict_image(model: tf.keras.Model, input_batch: np.ndarray) -> dict:
    probabilities = model.predict(input_batch, verbose=0)[0].astype(float)
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    probability_table = pd.DataFrame(
        {
            "Class": [DISPLAY_NAMES[name] for name in CLASS_NAMES],
            "Probability": probabilities,
        }
    ).sort_values("Probability", ascending=False, ignore_index=True)

    return {
        "predicted_index": predicted_index,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "table": probability_table,
    }

def confidence_label(confidence: float) -> str:
    if confidence >= 0.85:
        return "High confidence"
    if confidence >= 0.65:
        return "Moderate confidence"
    return "Low confidence — review recommended"

with st.sidebar:
    st.header("Model Information")
    st.markdown(
        "**Architecture:** Fine-Tuned MobileNetV2  \n"
        "**Input size:** 224 × 224 × 3  \n"
        "**Classes:** 6  \n"
        "**Test accuracy:** 92.17%  \n"
        "**Macro F1-score:** 92.35%"
    )
    st.divider()
    st.subheader("Supported Classes")
    for name in CLASS_NAMES:
        st.write(f"{CLASS_ICONS[name]} {DISPLAY_NAMES[name]}")
    st.divider()
    st.caption(
        "Confidence is the model's Softmax probability for the selected class "
        "and should not be interpreted as a guarantee of correctness."
    )

st.markdown(
    '''
    <div class="hero">
        <h1>Natural Scene Classifier</h1>
        <p>Upload a natural-scene image and classify it using the final fine-tuned MobileNetV2 model.</p>
    </div>
    ''',
    unsafe_allow_html=True,
)

try:
    with st.spinner("Loading the trained model..."):
        model = load_scene_model(str(MODEL_PATH))
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.code(str(exc))
    st.info(
        f"Place `{MODEL_FILENAME}` in the same folder as `app.py`, "
        "then restart the Streamlit application."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a natural-scene image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP.",
)

if uploaded_file is None:
    st.info("Upload an image to generate a scene prediction and probability distribution.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Final Model", "MobileNetV2")
    c2.metric("Test Accuracy", "92.17%")
    c3.metric("Macro F1", "92.35%")
else:
    try:
        uploaded_image = Image.open(uploaded_file)
        original_image, input_batch = prepare_image(uploaded_image)
        with st.spinner("Analyzing image..."):
            result = predict_image(model, input_batch)
    except UnidentifiedImageError:
        st.error("The uploaded file could not be decoded as a valid image.")
        st.stop()
    except Exception as exc:
        st.error("Prediction failed.")
        st.code(str(exc))
        st.stop()

    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    display_class = DISPLAY_NAMES[predicted_class]
    icon = CLASS_ICONS[predicted_class]

    image_col, result_col = st.columns([1.05, 0.95], gap="large")

    with image_col:
        st.subheader("Uploaded Image")
        st.image(original_image, caption=uploaded_file.name, use_container_width=True)

    with result_col:
        st.subheader("Prediction")
        st.markdown(
            f'''
            <div class="prediction-card">
                <div class="prediction-label">Predicted scene</div>
                <div class="prediction-value">{icon} {display_class}</div>
                <div class="prediction-label">Prediction confidence</div>
                <div class="confidence-value">{confidence:.2%}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.progress(min(max(confidence, 0.0), 1.0), text=confidence_label(confidence))
        st.caption("Confidence represents the Softmax probability assigned to the predicted class.")

    st.divider()
    st.subheader("Class Probability Distribution")

    probability_df = result["table"].copy()
    probability_df["Probability (%)"] = probability_df["Probability"] * 100
    chart_df = probability_df.set_index("Class")[["Probability (%)"]]
    st.bar_chart(chart_df, horizontal=True)

    display_df = probability_df[["Class", "Probability"]].copy()
    display_df["Probability"] = display_df["Probability"].map(lambda x: f"{x:.2%}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()
    with st.expander("Technical inference details"):
        st.write(f"**Model file:** `{MODEL_FILENAME}`")
        st.write(f"**Input tensor shape:** `{input_batch.shape}`")
        st.write("**Input color mode:** RGB")
        st.write(
            "**External preprocessing:** None — the exported model already contains "
            "the MobileNetV2 rescaling layer."
        )
        st.write(f"**Predicted class index:** {result['predicted_index']}")

st.caption(
    "Computer Vision Project • Six-Class Natural Scene Classification • Fine-Tuned MobileNetV2"
)
