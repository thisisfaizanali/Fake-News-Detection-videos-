import streamlit as st
from moviepy.editor import VideoFileClip
import whisper
import os

def transcribe_video(video_file):
    cvt_video = VideoFileClip(video_file)
    ext_audio = cvt_video.audio
    audio_file="audio.mp3"
    ext_audio.write_audiofile(audio_file)

    model = whisper.load_model("base")
    result = model.transcribe(audio_file)

    os.remove(audio_file)

    return result["text"]

def main():
    st.title("Video Transcription")

    uploaded_file = st.file_uploader("Upload video file", type=["mp4"])

    if uploaded_file is not None:
        st.write("Video uploaded successfully!")
        st.write("Converting, Please wait.....")

        result = transcribe_video(uploaded_file.name)

        st.header("Transcription Result")
        st.text_area("Transcription", result, height=200)

        st.download_button(
            label="Download Transcript",
            data=result,
            file_name="transcript.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
