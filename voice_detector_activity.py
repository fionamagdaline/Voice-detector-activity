import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import wave
import time
import speech_recognition as sr

# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
THRESHOLD_ENERGY = 80000  # Adjust this value according to your environment
MIN_VOICE_DURATION = 0.5  # Minimum duration of voice segment in seconds
MAX_VOICE_DURATION = 5.0  # Maximum duration of voice segment in seconds
SAVE_PATH = "detected_voice.wav"

def plot_spectrum(fft_magnitude):
    plt.clf()
    plt.plot(np.arange(len(fft_magnitude)), fft_magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT Spectrum')
    plt.pause(0.01)

def detect_voice(fft_magnitude):
    peaks, _ = find_peaks(fft_magnitude, height=THRESHOLD_ENERGY, distance=100)
    if len(peaks) > 0:
        print("Voice detected!")
        return True
    else:
        return False

def record_voice(stream):
    frames = []
    start_time = None
    while True:
        data = stream.read(CHUNK)
        np_data = np.frombuffer(data, dtype=np.int16)
        fft_data = np.fft.fft(np_data)
        fft_magnitude = np.abs(fft_data)
        
        plot_spectrum(fft_magnitude)
        
        if detect_voice(fft_magnitude):
            frames.append(data)
            if start_time is None:
                start_time = time.time()
        elif start_time is not None:
            duration = time.time() - start_time
            if duration >= MIN_VOICE_DURATION:
                break
            else:
                frames = []
                start_time = None
    
    return frames

def save_voice(frames):
    # Write the recorded voice segment to a WAV file
    p = pyaudio.PyAudio()
    wf = wave.open(SAVE_PATH, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Use SpeechRecognition to transcribe the recorded voice segment
    recognizer = sr.Recognizer()
    with sr.AudioFile(SAVE_PATH) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print("Transcribed text:", text)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

def main():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* Recording")

    try:
        while True:
            frames = record_voice(stream)
            if frames:
                save_voice(frames)
                print("Voice segment saved to:", SAVE_PATH)

    except KeyboardInterrupt:
        print("* Stopped")

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    main()

