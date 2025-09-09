import os, json, yaml, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from dataset import load_datasets
from train_ser import DataCollator

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    out_dir = cfg["paths"]["output_dir"]
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(ckpt_dir)
    feat = Wav2Vec2FeatureExtractor.from_pretrained(ckpt_dir, sampling_rate=cfg["sample_rate"])
    _, _, ds_test = load_datasets(cfg)
    collator = DataCollator(feat)

    all_preds, all_labels = [], []
    for i in range(len(ds_test)):
        b = [ds_test[i]]
        batch = collator(b)
        with torch.no_grad():
            logits = model(**{k:v for k,v in batch.items() if k!="labels"}).logits
        pred = logits.argmax(-1).cpu().item()
        all_preds.append(pred); all_labels.append(b[0]["label"])

    labels = list(range(len(cfg["labels"])))
    cm = confusion_matrix(all_labels, all_preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cfg["labels"])
    plt.figure()
    disp.plot(xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix (Test)")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), bbox_inches="tight", dpi=200)

    report = classification_report(all_labels, all_preds, target_names=cfg["labels"], digits=4)
    print(report)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
