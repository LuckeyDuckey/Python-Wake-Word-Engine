import os, random, time, librosa
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

class data_processing:
    def __init__(self):
        self.features = []
        self.labels = []

    def trim(self, audio):
        amount = int((1.5 + random.randint(-100,0)/100) * 172.5)#86.25- 258.75
        new_audio = []

        for i in range(len(audio)):
            new_audio.append(audio[i][amount:amount+345])

        new_audio = np.array(new_audio)
        return new_audio

    def spec_augment(self, spec: np.ndarray, num_mask=2, 
                    freq_masking_max_percentage=0.05, time_masking_max_percentage=0.1):

        spec = spec.copy()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
            
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = -80#-80 / 0
            

            time_percentage = random.uniform(0.0, time_masking_max_percentage)
            
            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = -80 #-80 / 0
        
        return spec

    def bar(self, total, current):
        percent = 100 * (current / float(total))
        bar = "█" * int(percent) + "-" * (100 - int(percent))
        print(f"\r|{bar}| {percent:.2f}%", end="\r")

    def process_ww(self, clips_folder_path, save_path, up_sample):
        amount = int(len(os.listdir(clips_folder_path)))

        for i in range(amount):
                    
            try:
                self.bar(amount,i+1)
                
                audio, sr = librosa.load(clips_folder_path+str(i)+".wav")
                mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, hop_length=128, fmax=8000)
                ps_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

                trimmed = self.trim(ps_db)

                for i in range(up_sample):
                    augmented = self.spec_augment(trimmed)

                    self.features.append(augmented)
                    self.labels.append([1])

            except:
                print("File " + str(i) + ".wav does not work")

            #plt.imshow(augmented, origin="lower", cmap=plt.get_cmap("magma"))

    def process_bg(self, clips_folder_path, save_path):
        amount = int(len(os.listdir(clips_folder_path)))

        for i in range(amount):
                    
            try:
                self.bar(amount,i+1)
                
                audio, sr = librosa.load(clips_folder_path+str(i)+".wav")
                mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, hop_length=128, fmax=8000)
                ps_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

                self.features.append(ps_db)
                self.labels.append([0])

            except:
                print("File " + str(i) + ".wav does not work")

            #plt.imshow(augmented, origin="lower", cmap=plt.get_cmap("magma"))
    
    def process(self, ww_folder_path, bg_folder_path, save_path, up_sample):
        self.process_ww(ww_folder_path, save_path, up_sample)
        self.process_bg(bg_folder_path, save_path)
        np.save(save_path+"features.npy", self.features)
        np.save(save_path+"labels.npy", self.labels)

dp = data_processing()
dp.process(os.path.dirname(os.path.realpath(__file__)) + "\\Data\\Samples\\", os.path.dirname(os.path.realpath(__file__)) + "\\Data\\Bg\\", os.path.dirname(os.path.realpath(__file__)) + "\\Data\\", 1)
