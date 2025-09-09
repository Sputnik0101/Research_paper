import os, json, yaml, numpy as np, torch
from dataclasses import dataclass
from transformers import (
    Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)
from dataset import load_datasets

@dataclass
class DataCollator:
    feature_extractor: Wav2Vec2FeatureExtractor
    def __call__(self, batch):
        audio = [b["input_values"].numpy() for b in batch]
        inputs = self.feature_extractor(
            audio, sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt", padding=True
        )
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        inputs["labels"] = labels
        return inputs

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro"))
    }

def set_freeze_policy(model, unfreeze_last_n_layers=0):
    # freeze all
    for p in model.wav2vec2.parameters():
        p.requires_grad = False
    # unfreeze last N encoder layers
    if unfreeze_last_n_layers > 0:
        enc = model.wav2vec2.encoder.layers
        for layer in enc[-unfreeze_last_n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
    # always train the classification head
    for p in model.classifier.parameters():
        p.requires_grad = True

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    set_seed(cfg["seed"])
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)

    ds_train, ds_valid, ds_test = load_datasets(cfg)

    feat = Wav2Vec2FeatureExtractor.from_pretrained(cfg["model_name"], sampling_rate=cfg["sample_rate"])
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=len(cfg["labels"]),
        label2id={l:i for i,l in enumerate(cfg["labels"])},
        id2label={i:l for i,l in enumerate(cfg["labels"])}
    )

    # memory optimizations for 4GB GPU
    if cfg["train"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    if cfg["train"].get("freeze_encoder", True):
        set_freeze_policy(model, cfg["train"].get("unfreeze_last_n_layers", 0))

    collator = DataCollator(feature_extractor=feat)

    args = TrainingArguments(
      output_dir=os.path.join(cfg["paths"]["output_dir"], "checkpoints"),
      per_device_train_batch_size=cfg["train"]["batch_size"],
      per_device_eval_batch_size=cfg["train"]["batch_size"],
      gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
      evaluation_strategy="epoch",
      save_strategy="epoch",
      num_train_epochs=cfg["train"]["num_epochs"],
      learning_rate=cfg["train"]["learning_rate"],
      warmup_ratio=cfg["train"]["warmup_ratio"],
      fp16=cfg["train"]["fp16"],
      logging_steps=50,
      load_best_model_at_end=True,
      metric_for_best_model="macro_f1",
      greater_is_better=True,
      gradient_checkpointing=cfg["train"].get("gradient_checkpointing", True)
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=feat
    )

    trainer.train()
    eval_metrics = trainer.evaluate(ds_test)
    with open(os.path.join(cfg["paths"]["output_dir"], "metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print("Test metrics:", eval_metrics)

if __name__ == "__main__":
    main()
