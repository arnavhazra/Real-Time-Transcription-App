### Introduction

This repository contains a Streamlit application that uses the AssemblyAI API to perform real-time transcription. The app allows users to upload a lecture MP3 file, convert it to WAV format, preprocess the audio, and transcribe the uploaded audio in real-time.

### Setup

To run the project, you will need to install the following dependencies:

- Streamlit
- Websockets
- Asyncio
- Pyaudio
- Soundfile
- Pyloudnorm
- Noisereduce
- Pydub

You can install the required packages using the following command:

```bash
pip install streamlit websockets asyncio pyaudio soundfile pyloudnorm noisereduce pydub
```

### Usage

#### App.py

To use the Streamlit application, run the following command:

```bash
streamlit run app.py
```

This will open a web application in your browser where you can upload a lecture MP3 file, convert it to WAV format, preprocess the audio, and transcribe the uploaded audio in real-time.

### API Key

To use the AssemblyAI API, you will need to sign up for an API key. You can obtain an API key by following the instructions on the AssemblyAI website. Once you have obtained the API key, you can set it as an environment variable:

```bash
export API_KEY="your_assemblyai_api_key"
```

### Transcription Output

The transcription output will be displayed in the Streamlit app. You can also download the transcription output as a text file by clicking the "Download transcription" button.

### Cleanup

After the transcription is complete, the temporary files will be removed automatically. However, if you want to manually remove the temporary files, you can run the following commands:

```bash
rm preprocessed_lecture.wav
rm lecture.wav
rm transcription.txt
```

### Contributing

If you would like to contribute to this project, please submit a pull request with your proposed changes.

