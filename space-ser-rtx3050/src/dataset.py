import pandas as pd
import torch, torchaudio, yaml
from torch.utils.data import Dataset
from augmentation import SpaceCommsAugment

class AudioTable(Dataset):
    def __init__(self, csv_path, split, labels, sr=16000, augment_cfg=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"]==split].reset_index(drop=True)
        self.labels = labels
        self.label2id = {l:i for i,l in enumerate(labels)}
        self.id2label = {i:l for l,i in self.label2id.items()}
        self.sr = sr
        self.augment = SpaceCommsAugment(sr, augment_cfg) if augment_cfg else None

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav, sr = torchaudio.load(row["path"])
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # mono

        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        if self.augment and row["split"]=="train":
            wav = self.augment(wav)

        # normalize length to 5s (pad/trim) — adjust for your data
        target_len = int(self.sr * 5.0)
        if wav.shape[-1] < target_len:
            pad = target_len - wav.shape[-1]
            wav = torch.nn.functional.pad(wav, (0, pad))
        else:
            wav = wav[:, :target_len]

        label_id = self.label2id[row["label"]]
        return {"input_values": wav.squeeze(0), "label": label_id}

def load_datasets(cfg):
    ds_train = AudioTable(
        cfg["paths"]["metadata_csv"], "train", cfg["labels"], cfg["sample_rate"], cfg["augment"]
    )
    ds_valid = AudioTable(
        cfg["paths"]["metadata_csv"], "valid", cfg["labels"], cfg["sample_rate"], cfg["augment"]
    )
    ds_test = AudioTable(
        cfg["paths"]["metadata_csv"], "test", cfg["labels"], cfg["sample_rate"], cfg["augment"]
    )
    return ds_train, ds_valid, ds_test
