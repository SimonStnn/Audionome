import math
import os
import time
from typing import Final
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from model import predict, get_mfccs, get_chroma_features, get_tempo
from utils import generate_sample_dict, get_genres

DATASET_PATH: Final = os.path.join("dataset", "genres_original")
GENRES: Final = get_genres(DATASET_PATH)
GENRE_ROWS: Final = 3
GENRE_COLS: Final = math.ceil(len(GENRES) / GENRE_ROWS)

TABS_KEY: Final = "selected_tab"
TAB_STATE_UPLOAD: Final = 0
TAB_STATE_SELECT: Final = 1
TAB_STATE_INPUT: Final = 2

st.set_page_config(
    page_title="Audionome",
    page_icon="🎵",
    initial_sidebar_state="expanded",
)

st.title("Audionome")
st.text("Classify music into genres")

st.sidebar.title("Audionome")
st.sidebar.caption("Close the sidebar to view the main page")

# Display the supported genres
st.sidebar.header("Available genres")
st.sidebar.write("\n".join(f"1. **{genre.capitalize()}**" for genre in GENRES))


def get_selected_tab() -> int:
    return st.session_state[TABS_KEY]


def set_selected_tab(tab: int) -> None:
    st.session_state[TABS_KEY] = tab


if TABS_KEY not in st.session_state:
    set_selected_tab(TAB_STATE_UPLOAD)

st.text("Upload an audio sample, select a sample from the dataset, or record audio")

upload_tab, select_tab, input_tab = st.tabs(["Upload", "Select", "Input"])

with upload_tab:
    st.text("Upload an audio sample of 30 seconds.")
    st.info(
        "If the audio is longer than 30 seconds, please trim it first at [Audio Trimmer](https://audiotrimmer.com/)."
    )

    upload_file = st.file_uploader(
        label="Upload an audio sample",
        key="file_uploader",
        type=["mp3", "wav"],
        on_change=lambda: set_selected_tab(TAB_STATE_UPLOAD),
    )
    upload_file_name = upload_file.name if upload_file is not None else None

with select_tab:
    st.text("Select a sample from the dataset.")
    files: dict[str, str] = generate_sample_dict(
        DATASET_PATH,
        lambda x: f"{x.split('.', 1)[0].capitalize()} {int(x.split('.', 2)[1])}",
    )

    select_file: bytes | None = None
    select = st.selectbox(
        "Select a sample",
        list(files.keys()),
        index=None,
        on_change=lambda: set_selected_tab(TAB_STATE_SELECT),
    )
    if select is not None:
        selected_path = files[select]
        with open(selected_path, "rb") as f:
            select_file = f.read()
    select_file_name = select if select is not None else None

with input_tab:
    input_file = st.audio_input(
        "Record audio",
        on_change=lambda: set_selected_tab(TAB_STATE_INPUT),
    )
    input_file_name = "Recorded audio" if input_file is not None else None

file = (
    upload_file
    if get_selected_tab() == TAB_STATE_UPLOAD
    else select_file if get_selected_tab() == TAB_STATE_SELECT else input_file
)
file_name = (
    upload_file_name
    if get_selected_tab() == TAB_STATE_UPLOAD
    else select_file_name if get_selected_tab() == TAB_STATE_SELECT else input_file_name
)

if file is not None:
    st.write("File uploaded!")

    if isinstance(file, UploadedFile):
        raw_file = file.getvalue()
    else:
        raw_file = file

    # if not validate_audio_length(raw_file):
    #     raw_file = trim_audio(
    #         raw_file,
    #         start=0,
    #         end=MAX_AUDIO_LENGTH,
    #         file_format=file.name.split(".")[-1],
    #     )
    st.audio(raw_file)

    start_time = time.time()
    with st.spinner(f"Predicting genre for `{file_name}`..."):
        prediction = predict(raw_file)
    end_time = time.time()

    st.success(
        f"""
        Predicted genre for `{file_name}`: \n
        $\\textbf{{\\huge {prediction.capitalize()}}}$
        """,
        icon="🔥",
    )
    st.info(f"Prediction took **{end_time - start_time:.2f} seconds**")

    st.balloons()

    st.divider()

    # Interpreting the prediction
    st.header("Interpretation")

    with st.spinner("Calculating tempo..."):
        cols = st.columns(2, border=True)
        cols[0].metric("Prediction", prediction.capitalize())
        cols[1].metric("Tempo", f"{get_tempo(raw_file):.1f} BPM")

    st.header("Mel-Frequency Cepstral Coefficients (MFCCs)")
    st.write(
        """
        Mel-frequency cepstral coefficients (MFCCs) are a feature widely used in automatic speech and speaker recognition. They were introduced by Davis and Mermelstein in the 1980's, and have been state-of-the-art ever since.
        """
    )
    st.subheader(f"Prediction for `{file_name}`:")
    with st.spinner("Extracting MFCCs..."):
        st.pyplot(get_mfccs(raw_file))

    st.header("Chroma Features")
    st.write(
        """
        Chroma features are an interesting and powerful representation for music audio in which the entire spectrum is projected onto 12 bins representing the 12 distinct semitones (or chroma) of the musical octave.
        """
    )
    st.subheader(f"Prediction for `{file_name}`:")
    with st.spinner("Extracting chroma features..."):
        st.pyplot(get_chroma_features(raw_file))
