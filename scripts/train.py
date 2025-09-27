import argparse, numpy as np, pandas as pd, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, EarlyStoppingCallback)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1, zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/train.csv")
    ap.add_argument("--val_csv",   default="data/val.csv")
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--out_dir", default="models/bert_binary")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    print("CUDA available?", torch.cuda.is_available())

    # 读 CSV 为 datasets
    ds = load_dataset("csv", data_files={"train": args.train_csv, "validation": args.val_csv})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)
    ds = ds.map(tok, batched=True)

    # label 列名要求是 "label"，且是 int
    def cast_label(batch):
        batch["labels"] = [int(x) for x in batch["label"]]
        return batch
    ds = ds.map(cast_label, batched=True)

    keep = ["input_ids", "attention_mask", "labels"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),  # GPU 有就开 FP16
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    print("Validation:", trainer.evaluate())
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("Saved to:", args.out_dir)

if __name__ == "__main__":
    main()
