# Factory Email Extraction

This project parses a synthetic corpus of factory production emails (`data.txt`) and converts it into a structured CSV file (`output.csv`). It also includes a trained **ML model** (token‑level NER with scikit‑learn) that can automatically segregate email content and extract structured fields into a table.

## Overview

- **Input**: `data.txt`  
  A plain‑text file containing 100 emails.  
  Each email:
  - Is wrapped between `-----EMAIL START-----` and `-----EMAIL END-----`
  - Contains `To:`, `From:`, and `Subject:` headers
  - Has a free‑form body with embedded production details such as serial number, item name, description, specification, quantity, and deadline, phrased like natural emails but using consistent anchor phrases (e.g., `Serial Number SN-0001`, `item named "..."`, `item description is: ...`).

- **Output**: `output.csv`  
  A tabular representation where each email becomes a single row and both the raw text and extracted fields are available as columns.

## Prerequisites

- Python ≥ 3.14
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

Install dependencies:

```bash
uv pip install -e .
# or
pip install scikit-learn pandas tabulate
```

## Running the regex extraction script

1. Make sure `data.txt` (the email corpus) is in the project root.
2. Run:

   ```bash
   python data_processing.py
   ```

3. The script will create `output.csv` in the same directory and print the number of rows written.

## ML Model

The project includes a **Named Entity Recognition (NER)** model that learns to extract 6 structured fields from free‑form email text, without relying on regex rules.

### How it works

1. Email bodies are **tokenised** (with punctuation‑aware splitting).
2. Each token is assigned a **BIO tag** (e.g., `B-SERIAL`, `I-ITEM_NAME`, `O`).
3. Per‑token features are extracted: word shape, prefix/suffix, context window (±3 tokens), positional features.
4. A **LogisticRegression** classifier (scikit‑learn) is trained on these features.
5. At inference time, predicted BIO tags are decoded back into entity strings and inserted into a structured table.

### Extracted fields

| Field | Entity label | Example |
|---|---|---|
| `serial_number` | `SERIAL` | SN-0001 |
| `item_name` | `ITEM_NAME` | Gear Assembly Type A1 |
| `item_description` | `ITEM_DESC` | precision-cut gear assembly for the primary conveyor drive stage |
| `item_specification` | `ITEM_SPEC` | Material=Alloy Steel; Teeth=32; Module=2.5; Diameter=80mm; Finish=Heat Treated |
| `item_quantity` | `ITEM_QTY` | 150 |
| `item_deadline` | `ITEM_DEADLINE` | 2026-03-01 |

### Training the model

Train the model on the labelled dataset (`output.csv`):

```bash
# Train on the full dataset and save to model/
python train_model.py

# Train with a 80/20 evaluation split and print a classification report
python train_model.py --eval
```

The trained model is saved to the `model/` directory (`classifier.pkl`, `vectorizer.pkl`, `label_encoder.pkl`).

### Running predictions

Use `predict.py` to extract structured fields from emails using the trained model:

```bash
# Run on the full dataset (output.csv) and display a structured table
python predict.py

# Save extracted results to a CSV file
python predict.py --output segregated_output.csv

# Compare predictions against ground truth labels
python predict.py --compare

# Extract from a single email file
python predict.py --file path/to/email.txt

# Extract from stdin
echo "To: team@factory.example.com
From: planner@factory.example.com
Subject: New Order SN-9999

Hi, Serial Number SN-9999 is for the item named \"Widget X\". The item description is: a test widget. The item specification is: Size=10mm. The item quantity is 5 units, and the item deadline is 2026-12-01." | python predict.py --stdin
```

### Model performance

Evaluated with an 80/20 train‑test split:

| Metric | Value |
|---|---|
| Token‑level test accuracy | 99.93% |
| Field‑level exact match (all 6 fields) | 100.0% |

## `output.csv` structure

The CSV file is UTF‑8 encoded and uses a header row. Each row corresponds to exactly one email and contains the following columns:

- `source_email`  
  The full raw text of the email block, including headers and body (but without the `-----EMAIL START/END-----` markers). Useful if you want to re‑run different extraction logic later without going back to `data.txt`.

- `to`  
  The recipient email address parsed from the `To:` header (e.g., `production.team@factory.example.com`).

- `from`  
  The sender email address parsed from the `From:` header.

- `subject`  
  The email subject line parsed from the `Subject:` header (typically includes the serial number and a short description of the order).

- `body`  
  The email body text only. This is everything after the subject line inside the email block and contains the natural‑language description of the order.

- `serial_number`  
  The extracted production serial number, taken from patterns like `Serial Number SN-0001`.

- `item_name`  
  The extracted item name, taken from phrases such as `the item named "Gear Assembly Type A1"`.

- `item_description`  
  The extracted short description sentence, taken from `the item description is: ...`.

- `item_specification`  
  The extracted structured specification string, taken from `the item specification is: ...` (e.g., parameter list like `Material=SS304; Capacity=1000L; ...`).

- `item_quantity`  
  The numeric quantity requested in the order, parsed from clauses like `the required item quantity is 150 units`.

- `item_deadline`  
  The requested deadline date for the item, parsed from `the item deadline is YYYY-MM-DD`.

## Intended use

This dataset and extraction script are designed for:

- Prototyping **regex‑based information extraction** from semi‑structured emails.
- Building training and evaluation sets for **NLP models** that learn to extract entities (serial numbers, item names, specs, etc.) from email text.
- Demonstrating end‑to‑end workflows: unstructured text → parsing → CSV → downstream analytics in pandas or SQL.

Because both the raw email (`source_email`, `body`) and the extracted fields are present, you can easily iterate on your extraction rules, compare them to model predictions, or extend the schema with additional fields.

## Project structure

| File | Description |
|---|---|
| `data.txt` | Raw email corpus (100 emails) |
| `data_processing.py` | Regex‑based extraction script → `output.csv` |
| `output.csv` | Labelled dataset (raw text + extracted fields) |
| `train_model.py` | NER model training pipeline |
| `predict.py` | Inference script (model → structured table) |
| `model/` | Saved model artefacts (pickle files) |
| `segregated_output.csv` | ML‑extracted structured output |