import streamlit as st
import websockets
import asyncio
import base64
import json
import os
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, MediaStreamConstraints
import pyloudnorm as pyln
import librosa
from pydub import AudioSegment
import os.path
import numpy as np
import requests

os.environ["FFMPEG_BINARY"] = os.path.join(os.path.expanduser("~"), "trsc-updated", "venv", "bin", "ffmpeg")
os.environ["FFPROBE_BINARY"] = os.path.join(os.path.expanduser("~"), "trsc-updated", "venv", "bin", "ffprobe")


def custom_remix(y, intervals):
    y_out = np.zeros_like(y)
    for i, (start, end) in enumerate(intervals):
        y_out[i * (end - start): (i + 1) * (end - start)] = y[start:end]
    return y_out

# Session state
if 'text' not in st.session_state:
    st.session_state['text'] = 'Listening...'
    st.session_state['run'] = False

# Audio parameters
st.sidebar.header('Audio Parameters')

RATE = int(st.sidebar.text_input('Rate', 16000))

def transcribe_uploaded_audio(api_key, audio_url):
    import requests
    import time

    endpoint = "https://api.assemblyai.com/v2/transcript"
    headers = {"authorization": api_key}

    # Submit the transcription request
    response = requests.post(endpoint, json={"audio_url": audio_url}, headers=headers)
    transcript_id = response.json()["id"]

    # Poll the API for the transcription result
    while True:
        response = requests.get(f"{endpoint}/{transcript_id}", headers=headers)
        status = response.json()["status"]

        if status == "completed":
            return response.json()["text"]
        elif status == "error":
            raise Exception("Transcription failed")

        time.sleep(5)

# Toggle transcription
def toggle_transcription():
    st.session_state['run'] = not st.session_state['run']

# Web user interface
st.title('🎙️ Real-Time Transcription App')

with st.expander('About this App'):
    st.markdown('''
    This Streamlit app uses the AssemblyAI API to perform real-time transcription.
    ''')

toggle_button = st.button('Toggle Transcription', on_click=toggle_transcription)

if st.session_state['run']:
    st.write("Transcription is active.")
else:
    st.write("Transcription is inactive.")

# Capture audio using streamlit-webrtc
constraints = MediaStreamConstraints(audio=True, video=False)
audio_ctx = webrtc_streamer(key="audio", mode=WebRtcMode.SENDONLY, media_stream_constraints=constraints)

# File uploader
uploaded_file = st.file_uploader("Upload a lecture MP3 file", type=["mp3"])

# Convert the uploaded MP3 file to WAV format and preprocess the audio
if uploaded_file is not None:
    st.write("Transcription has started for the uploaded lecture")
    audio = AudioSegment.from_mp3(uploaded_file)
    audio.export("lecture.wav", format="wav")

    # Load the WAV file
    data, rate = sf.read("lecture.wav")

    # Convert stereo to mono
    data_mono = librosa.to_mono(data.T)

    # Loudness normalization
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data_mono)
    target_loudness = -16.0
    gain = target_loudness - loudness
    data_mono = pyln.normalize.loudness(data_mono, gain, target_loudness)

    # Noise reduction
    reduced_noise_data = custom_remix(data_mono, intervals=[(0, len(data_mono))])

    # Save the preprocessed audio
    sf.write("preprocessed_lecture.wav", reduced_noise_data, rate)

    # Upload the preprocessed audio to AssemblyAI
    with open("preprocessed_lecture.wav", "rb") as f:
        response = requests.post("https://api.assemblyai.com/v2/upload", headers={"authorization": st.secrets['api_key']}, data=f)
        audio_url = response.json()["upload_url"]

    # Transcribe the uploaded audio
    transcription = transcribe_uploaded_audio(st.secrets['api_key'], audio_url)
    st.write(transcription)

    # Remove the temporary files
    os.remove("preprocessed_lecture.wav")
    os.remove("lecture.wav")

# Send audio (Input) / Receive transcription (Output)
async def send_receive():
    URL = f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate={RATE}"

    async with websockets.connect(
        URL,
        extra_headers=(("Authorization", st.secrets['api_key']),),
        ping_interval=5,
        ping_timeout=20
    ) as _ws:

        await asyncio.sleep(0.1)

        session_begins = await _ws.recv()

        async def send():
            while st.session_state['run']:
                try:
                    if audio_ctx.audio_receiver:
                        audio_frames = audio_ctx.audio_receiver.get_frames()
                    elif os.path.isfile("preprocessed_lecture.wav"):
                        audio_frames, _ = sf.read("preprocessed_lecture.wav")
                    else:
                        audio_frames = []

                    for frame in audio_frames:
                        data = frame.tobytes()
                        data = base64.b64encode(data).decode("utf-8")
                        json_data = json.dumps({"audio_data": str(data)})
                        await _ws.send(json_data)

                except websockets.exceptions.ConnectionClosedError as e:
                    break

                except Exception as e:
                    break

                await asyncio.sleep(0.01)

        async def receive():
            while st.session_state['run']:
                try:
                    result_str = await _ws.recv()
                    result = json.loads(result_str)['text']

                    if json.loads(result_str)['message_type'] == 'FinalTranscript':
                        st.session_state['text'] = result
                        st.write(st.session_state['text'])

                        with open('transcription.txt', 'a') as transcription_txt:
                            transcription_txt.write(st.session_state['text'])
                            transcription_txt.write(' ')

                except websockets.exceptions.ConnectionClosedError as e:
                    break

                except Exception as e:
                    break

        send_result, receive_result = await asyncio.gather(send(), receive())


def download_transcription():
    with open('transcription.txt', 'r') as read_txt:
        st.download_button(
            label="Download transcription",
            data=read_txt,
            file_name='transcription_output.txt',
            mime='text/plain')
        
try:
    if st.session_state['run']:
        asyncio.run(send_receive())

    if os.path.isfile('transcription.txt'):
        st.markdown('### Download')
        download_transcription()
finally:
    if os.path.isfile('transcription.txt'):
        os.remove('transcription.txt')

    if os.path.isfile("preprocessed_lecture.wav"):
        os.remove("preprocessed_lecture.wav")

    if os.path.isfile("lecture.wav"):
        os.remove("lecture.wav")