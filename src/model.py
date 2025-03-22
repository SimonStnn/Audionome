import os
import io
import pickle
import librosa as lr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Load the scaler
scaler = pickle.load(open(os.path.join(MODEL_DIR, "standard_scaler.sav"), "rb"))

# Load the model
model_name = os.path.join(MODEL_DIR, "svm_final_model.sav")
model = pickle.load(open(model_name, "rb"))

# Load the label encoder
label_encoder = pickle.load(open(os.path.join(MODEL_DIR, "label_encoder.sav"), "rb"))


def predict(audio_sample: bytes):
    """
    Predicts the genre of the given audio sample.
    """

    audio_file = io.BytesIO(audio_sample)
    y, sr = lr.load(audio_file)

    # Chroma STFT
    chroma_stft = lr.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)

    # RMS (Root Mean Square)
    rms = lr.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)

    # Spectral Centroid
    spectral_centroid = lr.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)

    # Spectral Bandwidth
    spectral_bandwidth = lr.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_var = np.var(spectral_bandwidth)

    # Spectral Rolloff
    rolloff = lr.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)

    # Zero Crossing Rate
    zero_crossing_rate = lr.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    zero_crossing_rate_var = np.var(zero_crossing_rate)

    # Harmony and Percussive components
    harmony, perceptr = lr.effects.hpss(y)
    harmony_mean = np.mean(harmony)
    harmony_var = np.var(harmony)
    perceptr_mean = np.mean(perceptr)
    perceptr_var = np.var(perceptr)

    # Tempo
    onset_env = lr.onset.onset_strength(y=y, sr=sr)
    tempo = lr.beat.tempo(onset_envelope=onset_env, sr=sr)

    # MFCC (Mel-Frequency Cepstral Coefficients)
    mfccs = lr.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    mfcc_means = []
    mfcc_vars = []
    for i in range(1, 21):  # MFCC 1-20
        mfcc_means.append(np.mean(mfccs[i - 1]))
        mfcc_vars.append(np.var(mfccs[i - 1]))

    # Extract features
    features = {
        "chroma_stft_mean": chroma_stft_mean,
        "chroma_stft_var": chroma_stft_var,
        "rms_mean": rms_mean,
        "rms_var": rms_var,
        "spectral_centroid_mean": spectral_centroid_mean,
        "spectral_centroid_var": spectral_centroid_var,
        "spectral_bandwidth_mean": spectral_bandwidth_mean,
        "spectral_bandwidth_var": spectral_bandwidth_var,
        "rolloff_mean": rolloff_mean,
        "rolloff_var": rolloff_var,
        "zero_crossing_rate_mean": zero_crossing_rate_mean,
        "zero_crossing_rate_var": zero_crossing_rate_var,
        "harmony_mean": harmony_mean,
        "harmony_var": harmony_var,
        "perceptr_mean": perceptr_mean,
        "perceptr_var": perceptr_var,
        "tempo": tempo,
    }

    # Add MFCC features
    for i in range(1, 21):
        features[f"mfcc{i}_mean"] = mfcc_means[i - 1]
        features[f"mfcc{i}_var"] = mfcc_vars[i - 1]

    # Create a DataFrame
    features_df = pd.DataFrame([features])

    features_df_scaled = scaler.transform(features_df)

    # Predict the genre
    prediction = model.predict(features_df_scaled)
    predicted_genre = label_encoder.inverse_transform(prediction)[0]
    print(f"The predicted genre is: {predicted_genre}")
    return str(predicted_genre)


def get_mfccs(content: bytes) -> plt.Figure:
    """
    Extracts the Mel-Frequency Cepstral Coefficients (MFCC) from the given audio sample.
    """
    y, sr = lr.load(io.BytesIO(content))
    mfccs = lr.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_normalized = lr.util.normalize(mfccs, axis=1)
    fig = plt.figure(figsize=(14, 5))
    spec = lr.display.specshow(mfccs_normalized, x_axis="time")
    fig.colorbar(spec)
    plt.ylabel("MFCC Coefficients")
    return fig
