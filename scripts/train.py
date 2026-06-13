"""Fine-tune BERT for sentiment classification on the IMDb dataset."""

from pathlib import Path
import sys

from datasets import load_dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text_classification.metrics import compute_metrics

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = ROOT / "outputs" / "results"
TRAIN_SAMPLE_SIZE = 2000


def main():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset("imdb", split=f"train[:{TRAIN_SAMPLE_SIZE}]")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()
