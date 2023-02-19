import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('saved_model/my_model')

groups = ['train','test']
labels = ['hotdog','nothotdog']

example = "hotdog.jpeg"

def predict_image(file):
    img = file
    tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32)
    tensor = tf.image.resize(tensor, [128, 128])
    input_tensor = tf.expand_dims(tensor, axis=0)
    predictions = model.predict(input_tensor)
    return labels[np.argmax(predictions)]

st.title("Hotdog or not Hotdog?")

uploaded_file = st.file_uploader("Upload your image to see if it's a hotdog: ")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    result = predict_image(bytes_data)
    if result == 'hotdog':
        is_hotdog = Image.open('yes.png')
    elif result == 'nothotdog':
        is_hotdog = Image.open('no.png')
    st.image(is_hotdog, width=400)
    st.image(uploaded_file, width=400)

url = "https://github.com/erkanncelen"
st.caption("Developed By: [Erkan Celen](%s)" % url)
