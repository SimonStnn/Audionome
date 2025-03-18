import time
import numpy as np
import pandas as pd
import streamlit as st

from model import predict
from utils import MAX_AUDIO_LENGTH, trim_audio, validate_audio_length


def main() -> None:
    st.title("Audionome")
    st.caption("Classify music into genres")
    st.write(
        """
        Upload a music file and we'll classify it into a genre! 
        """
    )

    file = st.file_uploader(
        label="Upluad an audio sample",
        key="file_uploader",
        type=["mp3", "wav"],
    )

    if file is not None:
        st.write("File uploaded!")

        raw_file = file.read()
        print(file.name)
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
        st.success(f"Prediction: **{prediction}**")
        st.info(f"Prediction took {end_time - start_time:.2f} seconds")

    dataframe = pd.DataFrame(
        np.random.randn(10, 20), columns=["col %d" % i for i in range(20)]
    )
    st.dataframe(dataframe.style.highlight_max(axis=0))

    df = pd.DataFrame({"first column": [1, 2, 3, 4], "second column": [10, 20, 30, 40]})

    option = st.selectbox("Which number do you like best?", df["first column"])

    "You selected: ", option

    left_column, right_column = st.columns(2)
    # You can use a column just like st.sidebar:
    left_column.button("Press me!")

    # Or even better, call Streamlit functions inside a "with" block:
    with right_column:
        chosen = st.radio(
            "Sorting hat", ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin")
        )
        st.write(f"You are in {chosen} house!")

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])

    st.header("Choose a datapoint color")
    color = st.color_picker("Color", "#FF0000")
    st.divider()
    st.scatter_chart(st.session_state.df, x="x", y="y", color=color)


if __name__ == "__main__":
    main()
