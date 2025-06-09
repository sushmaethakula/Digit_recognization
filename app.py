import streamlit as st
import cv2
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model
import numpy as np


drawing_mode = st.sidebar.selectbox("Drawing tool:", ("freedraw", "line", "rect", "circle", "transform"))
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 10)
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#000000")  # black
bg_color = st.sidebar.color_picker("Background color hex: ", "#FFFFFF")  # white
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Update in realtime", True)

@st.cache_resource
def load_mnist_model():
    return load_model("mnist_model.keras")

model = load_mnist_model()

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=280,
    width=280,
    drawing_mode=drawing_mode,
    key="canvas",
)

if canvas_result.image_data is not None:
    st.image(canvas_result.image_data, caption="Original Drawing")
    img = cv2.cvtColor(canvas_result.image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)
    img = 255 - img
    img_resized = cv2.resize(img, (28, 28))
    img_normalized = img_resized / 255.0
    final_img = img_normalized.reshape(1, 28, 28, 1)
    st.image(img_resized, caption="Preprocessed (28x28)")
    prediction = model.predict(final_img)
    st.write("Prediction:", np.argmax(prediction))
