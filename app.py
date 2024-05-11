import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

# Load your trained model and tokenizer
max_length = 35 # Define your max_length

model = tf.keras.models.load_model('model.h5') 
    
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load features from a pickle file
with open('features.pkl', 'rb') as file:
    features = pickle.load(file)

# Define the functions
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length, features):
    feature = features[image]
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

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = load_img(uploaded_file, target_size=(224,224))
    img = img_to_array(img)
    img = img/255.

    # Predict caption using your defined function
    predicted_caption = predict_caption(model, img, tokenizer, max_length, features)

    st.success(f"Predicted Caption: {predicted_caption}")