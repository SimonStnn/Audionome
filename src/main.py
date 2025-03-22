import io
import math
import os
import time
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from model import predict, get_mfccs
from utils import (
    MAX_AUDIO_LENGTH,
    generate_sample_dict,
    trim_audio,
    validate_audio_length,
    get_genres,
)

DATASET_PATH = os.path.join("dataset", "genres_original")


def main() -> None:
    st.title("Audionome")
    st.caption("Classify music into genres")

    st.write("Available genres:")

    # Display the supported genres
    genres = get_genres(DATASET_PATH)
    GENRE_ROWS = 3
    GENRE_COLS = math.ceil(len(genres) / GENRE_ROWS)

    # Split genres into columns and display
    for col, i in zip(st.columns(GENRE_COLS), range(0, len(genres), GENRE_ROWS)):
        txt: list[str] = []
        for j, genre in enumerate(genres[i : i + GENRE_ROWS]):
            txt.append(f"{i + j + 1}. {genre.capitalize()}")
        col.write("\n".join(txt))

    st.write("###")

    upload_tab, select_tab, input_tab = st.tabs(["Upload", "Select", "Input"])

    with upload_tab:
        upload_file = st.file_uploader(
            label="Upload an audio sample",
            key="file_uploader",
            type=["mp3", "wav"],
            on_change=lambda: print("now"),
        )
        upload_file_name = upload_file.name if upload_file is not None else None

    with select_tab:
        files: dict[str, str] = generate_sample_dict(
            DATASET_PATH,
            lambda x: f"{x.split('.', 1)[0].capitalize()} {int(x.split('.', 2)[1])}",
        )

        select_file: bytes | None = None
        select = st.selectbox(
            "Select a sample",
            list(files.keys()),
            index=None,
        )
        if select is not None:
            selected_path = files[select]
            with open(selected_path, "rb") as f:
                select_file = f.read()
        select_file_name = select if select is not None else None

    with input_tab:
        input_file = st.audio_input("Record audio")
        input_file_name = "input" if input_file is not None else None

    file = upload_file or select_file or input_file
    file_name = upload_file_name or select_file_name or input_file_name

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
        st.info(f"Prediction took {end_time - start_time:.2f} seconds")

        st.balloons()

        # Interpreting the prediction
        st.write(
            f"""
            ## Interpretation
            - **{prediction.capitalize()}** is the predicted genre.
            - The model predicted this genre with a confidence of **{100:.2f}%** TODO.
            - The prediction took **{end_time - start_time:.2f} seconds**.
            """
        )

        st.write("## MFCCs")
        st.write(
            """
            Mel-frequency cepstral coefficients (MFCCs) are a feature widely used in automatic speech and speaker recognition. They were introduced by Davis and Mermelstein in the 1980's, and have been state-of-the-art ever since.
            """
        )
        st.write(f"### Prediction for **{file_name}**:")
        st.pyplot(get_mfccs(raw_file))


if __name__ == "__main__":
    main()
