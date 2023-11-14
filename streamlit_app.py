import os
import random
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from build_models import create_cnn4


model = create_cnn4(32, 0.1, 0.001,224)
model.load_model('/Model/CNN4_best')
nowildfire = '/Figures/nowildfire' 
wildfire = '/Figures/nowildfire'


def process_image(image):
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

def random_select(num_images):
    wild = [image for image in os.listdir(wildfire)]
    nowild = [image for image in os.listdir(nowildfire)]
    selected_wild = random.sample(wild, num_images)
    selected_nowild = random.sample(nowild, num_images)

    return selected_nowild + selected_wild

def display_images():
    images = random_select(2)
    


def main():
    st.title('Wildfire Detection')

    st.write('''
             Wildfires are destructive and fast-spreading fires 
             that can devastate forests, grasslands, and communities. 
             Detecting wildfires early is crucial for mitgating their 
             impact and ensuring public safety. This wildfire detection 
             model leverages cutting-edge technology to identify and 
             alert authorities about potential wildfires, helping to 
             combat these natural disasters and protect our 
             environment and communities.''')
    
