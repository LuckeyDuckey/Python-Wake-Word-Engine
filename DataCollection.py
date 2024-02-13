import sounddevice as sd
from scipy.io.wavfile import write
import os

import time

def record_sample():
    
    fs = 44100
    seconds = 4
    
    input("To start audio recording press Enter: ")
    
    for i in range(100):
        
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        time.sleep(1.2)
        print("Say Jarvis")
        sd.wait()
        
        write(os.path.dirname(os.path.realpath(__file__)) + "\\Data\\Samples\\" + str(i) + ".wav", fs, myrecording)
        input(f"Press to record next or two stop press ctrl + C ({i + 1})")


record_sample()

def record_background():
    
    fs = 44100
    seconds = 2
    
    input("To start audio recording press Enter: ")
    
    for i in range(500):
        
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        
        write(os.path.dirname(os.path.realpath(__file__)) + "\\Data\\Bg\\" + str(i) + ".wav", fs, myrecording)
        print(f"Recording ({i + 1})")


record_background()

