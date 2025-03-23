import os
from typing import Final

import streamlit as st

from model import get_mfccs

st.set_page_config(
    page_title="Audionome - MFCCs",
    page_icon="🖥️",
)

SAMPLES_PATH: Final = os.path.join("dataset", "genres_original")
JASS_SAMPLE_PATH: Final = os.path.join(SAMPLES_PATH, "jazz", "jazz.00020.wav")
POP_SAMPLE_PATH: Final = os.path.join(SAMPLES_PATH, "pop", "pop.00020.wav")
ROCK_SAMPLE_PATH: Final = os.path.join(SAMPLES_PATH, "rock", "rock.00020.wav")


st.title("MFCCs")
st.text("Extract Mel-frequency cepstral coefficients (MFCCs) from audio samples")

st.header("What are MFCCs?")
st.write(
    """
    Mel-frequency cepstral coefficients (MFCCs) are a feature extraction technique used in speech and audio processing.
    They are derived from the short-time Fourier transform (STFT) of an audio signal and are used to represent the spectral envelope of the signal.
    MFCCs are commonly used in speech recognition and music information retrieval tasks.
    """
)

st.header("How are MFCCs computed?")
st.write(
    """
    The computation of MFCCs involves several steps:

    1. Pre-emphasis: The audio signal is passed through a high-pass filter to amplify high-frequency components.
    2. Framing: The signal is divided into short overlapping frames.
    3. Windowing: Each frame is multiplied by a window function to reduce spectral leakage.
    4. Fourier transform: The Fourier transform is applied to each frame to obtain the frequency spectrum.
    5. Mel filterbank: The frequency spectrum is passed through a bank of triangular filters spaced on the mel scale.
    6. Logarithm: The log of the filterbank energies is taken.
    7. Discrete cosine transform (DCT): The DCT is applied to the log filterbank energies to obtain the MFCCs.
    """
)

st.header("Extracting MFCCs")

st.subheader("Jazz Music Sample")
with st.spinner("Extracting MFCCs..."):
    with open(JASS_SAMPLE_PATH, "rb") as f:
        jazz_sample = f.read()
    jazz_mfccs = get_mfccs(jazz_sample)
    st.pyplot(jazz_mfccs)

st.subheader("Pop Music Sample")
with st.spinner("Extracting MFCCs..."):
    with open(POP_SAMPLE_PATH, "rb") as f:
        pop_sample = f.read()
    pop_mfccs = get_mfccs(pop_sample)
    st.pyplot(pop_mfccs)

st.subheader("Rock Music Sample")
with st.spinner("Extracting MFCCs..."):
    with open(ROCK_SAMPLE_PATH, "rb") as f:
        rock_sample = f.read()
    rock_mfccs = get_mfccs(rock_sample)
    st.pyplot(rock_mfccs)
