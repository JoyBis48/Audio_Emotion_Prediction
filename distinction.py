 # type: ignore

# Importing the required libraries
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Function to create the gender classification model
def create_model(vector_length=128):
    model = Sequential([
    Dense(256, input_shape=(vector_length,), activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()
    return model

# Loading the pre-trained model
model = create_model()
model.load_weights("distinction_weights.h5")

# Function to extract features from audio file
def extract_feature(file_name):
    X, sample_rate = librosa.core.load(file_name)
    result = np.array([])
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
    result = np.hstack((result, mel))
    return result

# Function to classify gender
def classify_gender(file_path):
    features = extract_feature(file_path).reshape(1, -1)
    male_prob = model.predict(features, verbose=0)[0][0]
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"
    probability = "{:.2f}".format(male_prob) if gender == "male" else "{:.2f}".format(female_prob)
    return gender, probability
