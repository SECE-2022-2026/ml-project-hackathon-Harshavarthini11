import pandas as pd
import numpy as np
import librosa
import sounddevice as sd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
import librosa.display

# Load the dataset
url = "voice.csv"
data = pd.read_csv(url)

# Check class distribution to identify any imbalance
st.write("Class Distribution:")
st.write(data['label'].value_counts())

# Prepare the data
X = data.iloc[:, :-1]
y = data['label'].apply(lambda x: 1 if x == 'male' else 0)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
st.write("\nClassification Report:")
st.write(classification_report(y_test, y_pred))

# Save the trained model and scaler
joblib.dump(model, "voice_gender_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Load the saved model and scaler
model = joblib.load("voice_gender_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI setup
st.title("Voice Gender Recognition")
st.write("Speak into your microphone, and the app will predict whether your voice is male or female.")

# Function to record audio
def record_audio(duration=5, sr=22050):
    st.write("Recording... Speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    st.write("Recording finished.")
    return np.squeeze(audio), sr

# Function to extract features from audio
def extract_features(audio, sr):
    st.write("Extracting features from audio...")
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    
    # Visualize the MFCCs for debugging
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title("MFCC")
    st.pyplot(plt)

    return np.mean(mfccs.T, axis=0)

# Record and predict when the button is pressed
if st.button("Record and Predict"):
    try:
        # Record the audio and extract features
        audio, sr = record_audio()
        features = extract_features(audio, sr)

        # Scale the extracted features
        features_scaled = scaler.transform([features])
        
        # Debugging: Check if the feature dimensions match the training set
        st.write("Extracted Features:", features)
        st.write("Feature Dimensions:", features_scaled.shape)

        if features_scaled.shape[1] == X_train.shape[1]:
            prediction = model.predict(features_scaled)
            st.write("Predicted Gender:", "Male" if prediction[0] == 1 else "Female")
        else:
            st.write("Error: Feature mismatch. Retrain the model with consistent feature extraction.")
    
    except Exception as e:
        st.write("Error:", e)