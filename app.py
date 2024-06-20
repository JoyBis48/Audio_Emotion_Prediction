# Importing Libraries
import librosa  
import numpy as np  
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from st_audiorec import st_audiorec

# Defining Emotion list based on encode classes
emotion_list = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Utliity Functions
def extract_features(audio_path):
    data, sr = librosa.load(audio_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=64)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

# Making the detection function
def detect(audio_path):
    features = extract_features(audio_path)
    features = np.expand_dims(features, axis=0) # Adding a batch dimension
    pred = model.predict(features) 
    pred_index = np.argmax(pred)
    label = emotion_list[pred_index]
    return label

# Load the pre-trained model
model = load_model('saved_model/model.keras')

# Streamlit app
st.title("Audio Emotion Prediction")
st.write("This application predicts the type of emotion from audio files using a Convolutional Neural Network")

# Option to upload a file
uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])

if uploaded_file is not None:
    with open("uploaded_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio(uploaded_file, format='audio/wav')
    if st.button("Submit"):
        emotion = detect("uploaded_audio.wav")
        st.write(f"Predicted Emotion: {emotion}")

# Recording audio
record_data = st_audiorec()



