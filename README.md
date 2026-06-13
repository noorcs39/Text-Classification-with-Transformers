# Text Classification with Transformers

<div align="center">

**Fine-tune BERT for movie review sentiment classification**

<p>
  <img src="https://img.shields.io/badge/Transformers-Hugging%20Face-059669?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/Model-BERT%20Base-0F766E?style=for-the-badge&logo=google&logoColor=white" alt="BERT" />
  <img src="https://img.shields.io/badge/Dataset-IMDb-334155?style=for-the-badge&logo=imdb&logoColor=white" alt="IMDb" />
  <img src="https://img.shields.io/badge/Python-3.x-14B8A6?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
</p>

<p>
  <img src="https://img.shields.io/badge/Task-Sentiment%20Classification-047857?style=flat-square" alt="Task" />
  <img src="https://img.shields.io/badge/Metrics-Accuracy%20%7C%20F1-64748B?style=flat-square" alt="Metrics" />
  <img src="https://img.shields.io/badge/License-MIT-10B981?style=flat-square" alt="MIT License" />
</p>

</div>

## Overview

**Text Classification with Transformers** is a compact NLP project that fine-tunes `bert-base-uncased` on the IMDb movie review dataset for binary sentiment classification. The workflow uses Hugging Face `Trainer`, tokenizes review text, and reports accuracy, precision, recall, and F1 score.

This repository is maintained by **Noor Uddin** as a practical example of transformer-based text classification.

## Features

- Loads the IMDb dataset through Hugging Face `datasets`.
- Tokenizes text with `BertTokenizer`.
- Fine-tunes `BertForSequenceClassification`.
- Tracks accuracy, precision, recall, and F1 during evaluation.
- Saves training artifacts under `outputs/results/`.
- Includes reusable metric helpers and unit tests.

## Project Structure

```text
Text-Classification-with-Transformers/
|-- scripts/
|   `-- train.py
|-- src/
|   `-- text_classification/
|       |-- __init__.py
|       `-- metrics.py
|-- tests/
|   `-- test_metrics.py
|-- outputs/
|   `-- .gitkeep
|-- requirements.txt
|-- LICENSE
|-- .gitignore
`-- README.md
```

## Setup

```bash
git clone https://github.com/noorcs39/Text-Classification-with-Transformers.git
cd Text-Classification-with-Transformers
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the training script:

```bash
python scripts/train.py
```

By default, the script:

- Uses the first **2000** IMDb training samples.
- Fine-tunes **BERT base uncased** for **3 epochs**.
- Writes checkpoints to `outputs/results/`.

## Model Details

| Setting | Value |
| --- | --- |
| Base model | `bert-base-uncased` |
| Task | Binary sentiment classification |
| Dataset | IMDb (`datasets`) |
| Learning rate | `2e-5` |
| Batch size | 16 train / 64 eval |
| Epochs | 3 |

## Testing

Run the lightweight unit tests:

```bash
python -m pytest
```

The tests validate the metric helper used by the Hugging Face `Trainer`. Full model training is not part of the automated test suite because it requires downloading BERT weights and significant runtime.

## Repository Topics

Suggested GitHub topics:

`transformers` `bert` `text-classification` `sentiment-analysis` `huggingface` `imdb` `nlp` `machine-learning` `deep-learning` `python`

## Contact

**Noor Uddin**
[noor.cs2@yahoo.com](mailto:noor.cs2@yahoo.com)

<p>
  <a href="https://github.com/noorcs39">
    <img src="https://img.shields.io/badge/GitHub-noorcs39-0F766E?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
  </a>
  <a href="mailto:noor.cs2@yahoo.com">
    <img src="https://img.shields.io/badge/Email-noor.cs2%40yahoo.com-14B8A6?style=for-the-badge&logo=yahoo&logoColor=white" alt="Email" />
  </a>
</p>

## License

This project is licensed under the [MIT License](LICENSE).
