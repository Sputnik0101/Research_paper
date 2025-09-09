import argparse, torch, torchaudio, yaml
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

def load_audio(path, sr=16000):
    wav, in_sr = torchaudio.load(path)
    if wav.shape[0]>1: wav = wav.mean(dim=0, keepdim=True)
    if in_sr != sr: wav = torchaudio.functional.resample(wav, in_sr, sr)
    target_len = int(sr*5.0)
    if wav.shape[-1] < target_len:
        pad = target_len - wav.shape[-1]
        wav = torch.nn.functional.pad(wav, (0,pad))
    else:
        wav = wav[:, :target_len]
    return wav.squeeze(0)

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    model_dir = "./outputs/checkpoints"
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
    feat = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, sampling_rate=cfg["sample_rate"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True, help="Path to wav file")
    args = parser.parse_args()

    wav = load_audio(args.wav, cfg["sample_rate"])
    inputs = feat([wav.numpy()], sampling_rate=cfg["sample_rate"], return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax(-1).item()
    label = model.config.id2label[pred_id]
    conf = torch.softmax(logits, dim=-1)[0, pred_id].item()
    print(f"Prediction: {label} (conf={conf:.2f})")

if __name__ == "__main__":
    main()
