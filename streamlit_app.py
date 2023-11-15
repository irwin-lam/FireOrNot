import os
from pathlib import Path
import random
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from src.build_models import create_cnn4

#Loading the model
model = create_cnn4(32, 0.1, 0.001,224)

#silencing errors
status = model.load_weights('./Models_weights/CNN4_best').expect_partial()

#Paths to figures selected
nowildfire = './Figures/nowildfire' 
wildfire = './Figures/wildfire'

#Labels
class_labels = ['Not Wildfire', 'Wildfire']

#Function to change the images to match the model
def process_image(image):
    try:
        resized_image = image.resize((224,224))
        image_array = np.array(resized_image)
        normalized_image = image_array/255.0
        input_data = np.expand_dims(normalized_image, axis = 0)
        return input_data
    except Exception as e:
        st.warning("Please upload a valid image")
        return None

#Function to output the prediction after running through the model
def make_prediction(input_data):
    try:
        prediction = (model.predict(input_data) > .5).astype(int)
        return prediction[0][0]
    except ValueError as e:
        st.warning("")
        return None
    
#combines two functions together to return the label of the prediction
def image_process(image_path):
    image = Image.open(image_path)
    open_image = process_image(image)
    prediction = make_prediction(open_image)
    return class_labels[prediction]

#randomly select num_images from the paths and shuffles them
def random_select(num_images):
    wild_images = [image for image in os.listdir(wildfire)]
    nowild_images = [image for image in os.listdir(nowildfire)]
    selected_wild = random.sample(wild_images, num_images)
    selected_nowild = random.sample(nowild_images, num_images)
    
    selected_images = [(str(Path(os.path.join(wildfire, image))), "Wildfire") 
                       for image in selected_wild] + [(str(Path(os.path.join(nowildfire, image))), "Not Wildfire") 
                                                      for image in selected_nowild]
    random.shuffle(selected_images)

    return selected_images

#default image selections
if 'selected_images' not in st.session_state:
    st.session_state.selected_images = random_select(2)

#default attribute for storing how many images displayed
if 'num_displayed_images' not in st.session_state:
    st.session_state.num_displayed_images = 4

#Displaying the images
def display_images():
    col = st.columns(4)
    for idx, image_info in enumerate(st.session_state.selected_images):
        image_path = image_info[0]
        true_label = image_info[1]

        if idx < st.session_state.num_displayed_images:
            with col[idx % 4]:
                st.image(image_path, caption=f"Image {idx+1}", use_column_width=True)

                if st.button(f'Detect Image {idx + 1}'):
                    with st.spinner(f'Detection in progress for Image {idx + 1}'):
                        prediction_label = image_process(image_path)
                        col[idx % 4].write(f'Prediction: {prediction_label}')
                        col[idx % 4].write(f'True: {true_label}')


#predicting the uploaded image        
def predict_image(uploaded_file):
    col = st.columns(1)
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with col[0]:
            if st.button('Detect'):
                with st.spinner('Model is working...'):
                    prediction = image_process(uploaded_file)
                    st.write(f"<div style='text-align: center;'>This Image Is {prediction}.</div>", unsafe_allow_html=True)
            st.write("")
            st.image(image, caption = 'Uploaded Image', use_column_width = True)
            
def main():
    st.markdown(
    "<div style='text-align: center;'>"
    "<h1 style='color: #FF4500;'>Wildfire Detection</h1>"
    "</div>",
    unsafe_allow_html=True
)

    st.write('''
             Wildfires are destructive and fast-spreading fires 
             that can devastate forests, grasslands, and communities. 
             Detecting wildfires early is crucial for mitgating their 
             impact and ensuring public safety. This wildfire detection 
             model leverages cutting-edge technology to identify and 
             alert authorities about potential wildfires, helping to 
             combat these natural disasters and protect our 
             environment and communities.''')
    
    st.write("")
    
    st.markdown(
    "<div style='text-align: center;'>"
    "<h5 style='color: #ff6138;'>Select how many images to display</h1>"
    "</div>",
    unsafe_allow_html=True
    )
    num_images_slider = st.slider("", 1, 10, st.session_state.num_displayed_images)

    #Updating the number of images displayed
    if num_images_slider != st.session_state.num_displayed_images:
        st.session_state.selected_images = random_select(int(np.ceil(num_images_slider/2)))
        st.session_state.num_displayed_images = num_images_slider

    display_images()
    
    st.write("")
    uploaded_file = st.file_uploader("Please upload an image...", type=['jpg', 'jpeg', 'png'])

    predict_image(uploaded_file)

if __name__ == "__main__":
    main()