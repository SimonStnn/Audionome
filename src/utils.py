import os
import io
import time
from pydub import AudioSegment

MAX_AUDIO_LENGTH = 30 * 1000  # 30 seconds in milliseconds
TMP_FOLDER = os.path.join("data", "tmp")


def validate_audio_length(audio_sample: bytes) -> bool:
    """Returns true if it is less than 30 seconds long."""
    return len(audio_sample) <= MAX_AUDIO_LENGTH


def trim_audio(audio_sample: bytes, start: int, end: int, file_format: str) -> bytes:
    """
    Trims the audio sample to the given start and end time.
    """
    # create dir if not exist
    os.makedirs(TMP_FOLDER, exist_ok=True)
    # store the audio to tmp folder
    tmp_file = os.path.join(TMP_FOLDER, f"{time.time()*1000000}.{file_format}")
    print(f"Saving audio to {tmp_file}")
    with open(tmp_file, "wb") as f:
        f.write(audio_sample)
    
    audio = AudioSegment.from_file(tmp_file, format=file_format)

    audio = audio[start:end]
    return audio.raw_data
