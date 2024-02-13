import pyaudio
from collections import deque
import threading
import time, os

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model(os.path.dirname(os.path.realpath(__file__)) + "\\Data\\WWD.h5")

class AudioStream(threading.Thread):
    def __init__(self):
        super(AudioStream, self).__init__()
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 22050
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk_size)
        self.queue = deque(maxlen=int(2 * self.rate / self.chunk_size))
        self.running = True
    
    def run(self):
        while self.running:
            data = self.stream.read(self.chunk_size)
            self.queue.append(data)
    
    def stop(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
    
    def get_recent_audio(self, seconds):
        audio_data = b''.join(self.queue)
        return np.frombuffer(audio_data, dtype=np.int16)

audio_stream = AudioStream()
audio_stream.start()

time.sleep(3)
print("Listening...")

while True:
    audio_data = librosa.util.buf_to_float(audio_stream.get_recent_audio(2))
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=22050, n_mels=256, hop_length=128, fmax=8000)
    out = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    audio = np.array(out)
    audio = audio.reshape(1,audio.shape[0],audio.shape[1],1)
    prediction = model.predict(audio, verbose=0)
    
    print(np.round_(prediction, decimals = 1))
    if np.round_(prediction, decimals = 2)[0][0] >= 0.9:
        print("Detected")
    
audio_stream.stop()
