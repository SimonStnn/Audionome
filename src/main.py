import math
import os
import time
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from model import predict
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
            txt.append(f"{i + j + 1}. {genre}")
        col.write("\n".join(txt))

    st.divider()

    st.write(
        """
        **Upload** a music file or **select** a sample and we'll classify it into a genre! 
        """
    )

    file: UploadedFile | bytes | None = None
    left_column, right_column = st.columns(2, border=True)
    with left_column:
        file = st.file_uploader(
            label="Upluad an audio sample",
            key="file_uploader",
            type=["mp3", "wav"],
            disabled=file is not None,
        )
    with right_column:
        # Grab 10 random files from dataset
        files: dict[str, str] = generate_sample_dict(
            DATASET_PATH, lambda x: f"{x.split('.', 1)[0]} {int(x.split('.', 2)[1])}"
        )

        select = st.selectbox(
            "Select a sample",
            list(files.keys()),
            index=None,
            disabled=file is not None,
        )

    if select is not None:
        selected_path = files[select]
        with open(selected_path, "rb") as f:
            file = f.read()

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
        with st.spinner("Processing..."):
            prediction = predict(raw_file)
        end_time = time.time()

        st.balloons()

        st.success(
            f"""
            Predicted genre: \n
            $\\textbf{{\\huge {prediction[0].upper() + prediction[1:]}}}$
            """,
            icon="🔥",
        )

        st.info(f"Prediction took {end_time - start_time:.2f} seconds")

    # dataframe = pd.DataFrame(
    #     np.random.randn(10, 20), columns=["col %d" % i for i in range(20)]
    # )
    # st.dataframe(dataframe.style.highlight_max(axis=0))


if __name__ == "__main__":
    main()
