import os, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

EMO_MAP_RAVDESS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

def parse_ravdess_name(name):
    # RAVDESS example: "03-01-05-02-02-02-12.wav"
    parts = name.split("-")
    if len(parts) < 3: return None
    emo_id = parts[2]
    return EMO_MAP_RAVDESS.get(emo_id)

def collect_audio_rows(root, labels):
    rows = []
    for p in Path(root).rglob("*.wav"):
        emo = None
        if "ravdess" in str(p).lower():
            emo = parse_ravdess_name(p.stem)
        # TODO: add IEMOCAP parser if you use it
        if emo is None: 
            continue
        if emo not in labels:
            continue
        rows.append({"path": str(p), "label": emo})
    return rows

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    data_root = cfg["paths"]["data_root"]
    labels = cfg["labels"]
    rows = collect_audio_rows(data_root, labels)
    assert len(rows) > 0, "No WAV files found. Place dataset under ./data/*"
    df = pd.DataFrame(rows)

    train_df, tmp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    valid_df, test_df = train_test_split(tmp, test_size=0.5, stratify=tmp["label"], random_state=42)

    out_csv = cfg["paths"]["metadata_csv"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    train_df["split"]="train"; valid_df["split"]="valid"; test_df["split"]="test"
    full = pd.concat([train_df, valid_df, test_df]).reset_index(drop=True)
    full.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with {len(full)} rows.")

if __name__ == "__main__":
    main()
