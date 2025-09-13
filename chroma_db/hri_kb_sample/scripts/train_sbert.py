# train_sbert.py
# Minimal example to fine-tune a bi-encoder on triples.jsonl using sentence-transformers.

import json, argparse, os
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample, SentenceTransformerTrainer, SentenceTransformerTrainingArguments

class TriplesDS(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path, 'r', encoding='utf-8')]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        return InputExample(texts=[r['anchor'], r['positive'], r['negative']])

def main(args):
    model = SentenceTransformer(args.model_name)
    ds = TriplesDS(args.triples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=True
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        loss=loss
    )
    trainer.train()
    model.save(out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default="data/triples.jsonl")
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out_dir", default="models/hri-bi-encoder")
    args = ap.parse_args()
    main(args)
