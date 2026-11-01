# Run this code in Google Colab with a T4 GPU.
# First run this in Colab:  !pip install -q transformers datasets accelerate scikit-learn

import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, f1_score, classification_report

ds = load_dataset("google-research-datasets/go_emotions", "simplified")
names = ds["train"].features["labels"].feature.names

GROUPS = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love",
            "optimism", "pride", "admiration", "desire", "relief", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}
to_group = {n: g for g, ns in GROUPS.items() for n in ns}
classes = list(GROUPS.keys())
c2i = {c: i for i, c in enumerate(classes)}


def keep(ex):
    return len({to_group[names[l]] for l in ex["labels"]}) == 1


def relabel(ex):
    ex["label"] = c2i[to_group[names[ex["labels"][0]]]]
    return ex


ds = ds.filter(keep).map(relabel)
print({k: len(v) for k, v in ds.items()})

MODEL = "google/electra-small-discriminator"
tok = AutoTokenizer.from_pretrained(MODEL)
tds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=64),
             batched=True, remove_columns=["text", "labels", "id"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(classes),
    id2label=dict(enumerate(classes)), label2id=c2i)


def metrics(p):
    pred = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, pred),
            "macro_f1": f1_score(p.label_ids, pred, average="macro")}


args = TrainingArguments(
    output_dir="out", num_train_epochs=4, per_device_train_batch_size=64,
    per_device_eval_batch_size=128, learning_rate=1e-4,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="macro_f1", fp16=True, report_to="none", save_total_limit=1)

trainer = Trainer(model=model, args=args, train_dataset=tds["train"],
                  eval_dataset=tds["validation"], processing_class=tok,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)

trainer.train()

out = trainer.predict(tds["test"])
print("TEST:", out.metrics)
print(classification_report(out.label_ids, out.predictions.argmax(-1), target_names=classes))

model.save_pretrained("sentiment_model")
tok.save_pretrained("sentiment_model")
