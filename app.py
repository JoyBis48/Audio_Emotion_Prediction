# Importing all the necessary libraries
import io
import librosa  
import numpy as np  
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from st_audiorec import st_audiorec
from scipy.io.wavfile import read as wav_read, write as wav_write 
import warnings
warnings.filterwarnings('ignore')
import joblib
from distinction import classify_gender

# Defining Emotion list based on encoded classes
emotion_list = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Utliity Functions
def extract_features(audio_path):
    # Loading the audio file
    data, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
    
    # Extracting features from the audio data
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=data, sr=sample_rate).T, axis=0)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=data, sr=sample_rate).T, axis=0)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
    poly_features = np.mean(librosa.feature.poly_features(y=data, sr=sample_rate).T, axis=0)

    # Horizontally stacking features
    features = np.hstack([zcr, chroma_stft, mfcc, rms, mel_spectrogram, spectral_contrast, tonnetz, spectral_rolloff, poly_features])

    return features

# Loading the scaler which was used to scale the features during training
scaler = joblib.load('scaler.pkl')  # assuming the scaler is saved in 'scaler.pkl'

# Making the detection function
def detect(audio_path):
    features = extract_features(audio_path)
    features = np.expand_dims(features, axis=0)  # Adding a batch dimension
    features = scaler.transform(features)  # Apply scaling
    pred = model.predict(features) 
    pred_index = np.argmax(pred)
    label = emotion_list[pred_index]
    return label

# Loading the pre-trained model
model = load_model('saved_model/model.keras')

# Streamlit app
st.title("Audio Emotion Prediction")
st.write("This application predicts the type of emotion from audio files using a Convolutional Neural Network")

# Providing option to upload a file
uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])

if uploaded_file is not None:
    with open("uploaded_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio(uploaded_file, format='audio/wav')
    if st.button("Submit"):
        gender, _ = classify_gender("uploaded_audio.wav")
        if gender=="male":
            st.write("Sorry :( , this model is only trained to detect emotions from female voices. Please upload female voice audio file.")
            os.remove("uploaded_audio.wav")
        else:
            emotion = detect("uploaded_audio.wav")
            st.write(f"Predicted Emotion: {emotion}")

# Providing option to record audio
recorded_data = st_audiorec()

if recorded_data is not None:
    # Converting byte string to numpy array
    wav_io = io.BytesIO(recorded_data)
    sr, audio_data = wav_read(wav_io)

    # Saving numpy array to WAV file
    wav_file_path = "recorded_audio.wav"
    wav_write(wav_file_path, sr, audio_data)

    st.audio(recorded_data, format='audio/wav')
    if gender=="male":
        st.write("Sorry :( , this model is only trained to detect emotions from female voices. Please upload female voice audio file.")
        os.remove("recorded_audio.wav")
    else:
        emotion = detect(wav_file_path)
        st.write(f"Predicted Emotion: {emotion}")





