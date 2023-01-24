import librosa
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

# ----------------------------
# Sound Dataset
# ----------------------------
class WhaleSoundDataSet(Dataset):
    def __init__(self, csv_path, data_path, configuration_dict):
        self.meta_df = None
        self.data_path = str(data_path)
        self.duration = configuration_dict.get('audio_duration', 2000)
        self.sr = configuration_dict.get('audio_sample_rate', 2000)
        self.channel = 2
        self.n_mels = configuration_dict.get('number_of_mel_filters', 64)

        self.meta_df = pd.read_csv(csv_path)
        
    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.meta_df)

    # ----------------------------

    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the filename
        audio_file_path = self.data_path + self.meta_df.loc[idx, 'clip_name']
        # soundData, sample_rate = torchaudio.load(filepath=audio_file_path, normalize = True)
        soundData, sr = librosa.load(audio_file_path, sr=2000, duration=2)
        # Get the Class ID, either 0 (no whale) or 1 (whale)
        # class_id = self.df.pop('label')
        # Get the Class ID
        class_id = self.meta_df.loc[idx, 'label']
        
        # This will convert audio files with two channels into one
        # soundData = torch.mean(soundData, dim=(0), keepdim=True)
        soundData_mono = librosa.to_mono(soundData)
        
        # Convert audio to log-scale Mel spectrogram
        # melspectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.resample, n_mels=self.n_mels)
        # melspectrogram = melspectrogram_transform(soundData)
        # melspectogram_db = torchaudio.transforms.AmplitudeToDB()(melspectrogram)

        # ------Librosa method -- may not be processing the data enough.
        # Load the audio data and sample rate from the AIFF file
        # data, sr = librosa.load(audio_file_path)
        # Compute the mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=soundData_mono, sr=sr)
        # ---------------- Librosa Data ^ ------------------------------
        
        
        
        return soundData_mono, sr, mel_spectrogram, class_id

