import os
import joblib
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf



def preprocess_image(image):
    try:
        resized_image = image.resize(224,224)
        image_array = np.array(resized_image)
        normalized_image = image_array/255.0
        input_data = np.expand_dims(normalized_image, axis = 0)
        return input_data
    except Exception as e:
        st.warning("Please upload a valid image")
        return None
    
def make_prediction(model, input_data):
    try:
        prediction = (model.predict(input_data) > .5).astype(int)
        return prediction
    except ValueError as e:
        st.warning("")
        return None
    
def main():
    pass