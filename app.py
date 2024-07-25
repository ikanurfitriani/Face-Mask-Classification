# MIT License

# Copyright (c) 2024 Ika Nurfitriani

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
from PIL import Image
import cv2
import streamlit as st
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("mask_classifier.h5")

def predict_image(image):
    try:
        # Convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize image to the size expected by the model
        image = cv2.resize(image, (128, 128))
        # Normalize the image
        image = image / 255.0
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Predict the class
        prediction = model.predict(image)
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Streamlit app
st.title('Face Mask Classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and convert the image to a format suitable for prediction
    image = Image.open(uploaded_file)
    image = image.convert("RGB")  # Ensure image is in RGB format
    image = np.array(image)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Predict and display results
    result = predict_image(image)
    if result is not None:
        class_idx = np.argmax(result)
        class_labels = ['Face With Mask', 'Face Without Mask']
        st.write(f"Prediction: {class_labels[class_idx]}")