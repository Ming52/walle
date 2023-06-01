import streamlit as st
import os
from fastai.vision.all import *

path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path, "export.pkl")

learn_inf = load_learner(model_path)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PILImage.create(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    pred, pred_idx, probs = learn_inf.predict(image)
    st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.04f}")
    