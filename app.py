# type: ignore
# Importing Libraries
import io
import librosa  
import numpy as np  
import os
from tensorflow.keras.models import load_model 
import streamlit as st
from st_audiorec import st_audiorec
from scipy.io.wavfile import read as wav_read, write as wav_write 
import random
from librosa.effects import time_stretch, pitch_shift
from sklearn.preprocessing import StandardScaler 

# Defining Emotion list based on encode classes
emotion_list = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Utliity Functions
 

def extract_features(data, sample_rate=22050):
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

    # Scaling features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.reshape(-1, 1))

    return scaled_features



# Define a mapping from class names to integers
class_map = {'Angry': 0, 'Disgusted': 1, 'Fearful': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprised': 6}

# Defining the noise transformation

def noise(data, noise_factor=1.0):
    noise_amp = 0.025*np.random.uniform()*np.amax(data) 
    data = data + noise_factor * noise_amp * np.random.normal(size=data.shape[0])  # adding random amount of gaussian noise for the entirety of the audio
    return data


# Making the generator function
def data_generator(files, batch_size=32):
    while True:
        # Shuffle the list of files
        random.shuffle(files)

        # Apply transformations to each file and stack them
        batch_data = []
        batch_labels = []
        for file in files:
            # Load the audio file
            data, sr = librosa.load(file, sr=22050, res_type='kaiser_fast')

            # Applying transformations with random intensities
            data = time_stretch(data, rate = random.uniform(0.5, 1.5) )
            data = pitch_shift(data, sr, n_steps = random.randint(-5, 5))
            data = noise(data, noise_factor = random.uniform(0, 1.5)) 

            # Extract features
            features = extract_features(data, sr=sr)

            # Get the label from the file name
            batch_data.append(features)
            label = os.path.basename(os.path.dirname(file))
            batch_labels.append(class_map[label])

            # Yield batches
            if len(batch_data) == batch_size:
                yield np.array(batch_data), np.array(batch_labels)
                batch_data = []
                batch_labels = []
        if batch_data:
            yield np.array(batch_data), np.array(batch_labels)


audio_dir = os.path.join(os.getcwd(), 'filtered_dataset')                
# Get a list of all audio files in the directory
all_files = []
for subdir, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.endswith(".wav"):
            all_files.append(os.path.join(subdir, file))

# Split the list of files into training and test sets
from sklearn.model_selection import train_test_split
train_files, test_files = train_test_split(all_files, test_size=0.2)

# Create generators for training and test sets
train_generator = data_generator(train_files)
test_generator = data_generator(test_files)


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
recorded_data = st_audiorec()

if recorded_data is not None:
    # Convert byte string to numpy array
    wav_io = io.BytesIO(recorded_data)
    sr, audio_data = wav_read(wav_io)

    # Save numpy array to WAV file
    wav_file_path = "recorded_audio.wav"
    wav_write(wav_file_path, sr, audio_data)

    st.audio(recorded_data, format='audio/wav')
    emotion = detect(wav_file_path)
    st.write(f"Predicted Emotion: {emotion}")





