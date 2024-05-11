import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.models import Model, load_model
import numpy as np
import pickle

# Load your trained model and tokenizer
max_length = 35 # Define your max_length

caption_model = load_model('model.h5') 

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the DenseNet201 model
fe = load_model('densenet201_model.h5')

# Define the functions
def generate_feature(image):
    img = img_to_array(image)
    img = img/255.
    img = np.expand_dims(img,axis=0)

    # Generate the feature
    feature = fe.predict(img, verbose=0)
    return feature

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, feature, tokenizer, max_length):
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = model.predict([feature,sequence])
        y_pred = np.argmax(y_pred)
        
        word = idx_to_word(y_pred, tokenizer)
        
        if word is None:
            break
            
        in_text += " " + word
        
        if word == 'endseq':
            break
            
    return in_text 

# Streamlit code
st.title("Image Captioning App")
st.write("Upload an image and get a caption!")

uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "png"])

max_length = 34

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = load_img(uploaded_file, target_size=(224,224))

    # Generate feature using your defined function
    feature = generate_feature(img)

    # Predict caption using your defined function
    predicted_caption = predict_caption(caption_model, feature, tokenizer, max_length)

    #remove startseq and endseq

    predicted_caption = predicted_caption.split()[1:-1]
    predicted_caption = ' '.join(predicted_caption)

    st.success(f"Predicted Caption: {predicted_caption}")