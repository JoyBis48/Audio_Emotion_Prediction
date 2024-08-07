# Audio Emotion Prediction

This project aims to detect emotions from audio files using a trained Convolutional Neural Network (CNN) model. It leverages the use of various audio features extracted using the **librosa** library. Streamlit is used for the deployment where it will ask the user to upload or record audio files and view the predicted emotions. The model currently predicts seven types of emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised. Additionally, it includes a gender classification model to ensure the emotion detection is performed only on female voices.

## Features

- **Emotion Detection**: Utilizes a pre-trained CNN model to predict emotions from uploaded or recorded audio files.
- **Gender Classification**: Ensures the emotion detection is performed only on female voices.
- **Graphical User Interface**: A user-friendly GUI built with Streamlit for easy interaction with the application.
- **Audio Preprocessing**: Audio files are preprocessed to extract relevant features for emotion detection.

## Installation

To run this code, you need to have Python installed on your system. The project has been tested on Python 3.12. Follow these steps below to set up the project:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/JoyBis48/Audio_Emotion_Prediction.git
    ```
2. **Navigate to the project directory**:
    ```sh
    cd Audio_Emotion_Prediction
    ```
3. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```
4. **To start the application, run the `app.py` script from the terminal**:
    ```sh
    streamlit run app.py
    ```

This will launch the streamlit site where you can upload or record audio files to detect emotions.

## How It Works

### Model Training

The application uses a model trained with TensorFlow on a dataset of audio files labeled with emotions. The training process involves extracting various audio features using the **librosa**library and training a CNN model to predict emotions based on these features.

### Audio Preprocessing

Uploaded or recorded audio files are preprocessed to extract relevant features that icludes zero-crossing rate, chroma STFT, MFCC, RMS, mel spectrogram, spectral contrast, tonnetz, spectral rolloff, and poly features. These features are then scaled using a pre-trained scaler before being fed into the emotion detection model.

### Emotion Prediction

The preprocessed audio features are fed into the trained model, which predicts the emotion displayed in the audio file. The model outputs a prediction corresponding to the emotions it has been trained on.

### GUI Interaction

Users interact with the application through a GUI built with Streamlit, uploading or recording audio files and viewing the predicted emotions. The GUI allows users to upload an audio file through the file uploader or record audio using the **st_audiorec** component.

## Dependencies

- TensorFlow
- Streamlit
- librosa
- numpy
- scipy
- joblib
- st_audiorec

## Files

- **app.py**: Main application script that runs the Streamlit GUI and handles audio file uploads and recordings.
- **distinction.py**: Contains the gender classification model and functions to classify gender from audio files.
- **requirements.txt**: Lists all the dependencies required to run the project.

## Usage

1. **Upload an Audio File**: Use the file uploader in the GUI to upload an audio file in WAV or MP3 format.
2. **Record Audio**: Use the audio recorder in the GUI to record audio directly.
3. **View Predicted Emotion**: The application will display the predicted emotion for the uploaded or recorded audio file.

## Note

The model is currently trained to detect emotions only from female voices. If a male voice is detected, the application will prompt the user to upload a female voice audio file.

