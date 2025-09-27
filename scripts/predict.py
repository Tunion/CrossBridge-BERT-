import argparse, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/bert_binary")
    ap.add_argument("--csv", default="data/val.csv")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--out", default="preds.csv")
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    enc = tok(df[args.text_col].tolist(), truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = logits.softmax(dim=-1).cpu().numpy()

    df["pred"] = probs.argmax(axis=-1)
    df["prob_abnormal"] = probs[:,1]
    df.to_csv(args.out, index=False)
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
