"""
Inference script: use the trained NER model to extract structured fields
from new (or existing) factory production emails and display them in a
structured table.

Usage:
    # Run on the original dataset and display extracted table
    python predict.py

    # Run on a single email from stdin
    echo "To: ..." | python predict.py --stdin

    # Run on a specific email text file
    python predict.py --file new_email.txt

    # Save results to CSV
    python predict.py --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from train_model import (
    ENTITY_FIELDS,
    MODEL_DIR,
    CSV_PATH,
    load_model,
    load_dataset,
    tokenize,
    token_features,
)

# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_tags(
    body: str,
    clf,
    vec,
    le,
) -> list[tuple[str, str]]:
    """Predict BIO tags for each token in *body*. Returns list of (token, tag)."""
    tokens = tokenize(body)
    if not tokens:
        return []

    feats = [token_features(tokens, i) for i in range(len(tokens))]
    X = vec.transform(feats)
    y_pred = clf.predict(X)
    tags = le.inverse_transform(y_pred)

    return [(tok[0], tag) for tok, tag in zip(tokens, tags)]


def _clean_entity(value: str, label: str) -> str:
    """Post-process an extracted entity string to remove boundary noise."""
    # Strip surrounding quotes and whitespace
    value = value.strip().strip('"').strip("'").strip()
    # Remove trailing punctuation that leaked in
    value = value.rstrip(".,;:!?")
    # Fix spacing around semicolons (tokenizer splits ; into separate token)
    value = re.sub(r"\s+;", ";", value)
    # For quantity, keep only digits
    if label == "ITEM_QTY":
        m = re.search(r"\d+", value)
        return m.group() if m else value
    # For deadline, keep only the date portion
    if label == "ITEM_DEADLINE":
        m = re.search(r"\d{4}-\d{2}-\d{2}", value)
        return m.group() if m else value.rstrip(".,;: ")
    return value


def extract_entities(token_tags: list[tuple[str, str]]) -> dict[str, str]:
    """Decode BIO tags into entity strings, one per entity type."""
    entities: dict[str, list[str]] = {}
    current_label: str | None = None

    for token, tag in token_tags:
        if tag.startswith("B-"):
            current_label = tag[2:]
            entities.setdefault(current_label, [])
            entities[current_label].append(token)
        elif tag.startswith("I-") and current_label == tag[2:]:
            entities[current_label].append(token)
        else:
            current_label = None

    # Map label → joined string, with cleanup
    result: dict[str, str] = {}
    label_to_field = {label: field for field, label in ENTITY_FIELDS}
    for label, toks in entities.items():
        field = label_to_field.get(label, label)
        raw = " ".join(toks)
        result[field] = _clean_entity(raw, label)

    return result


def predict_email(body: str, clf, vec, le) -> dict[str, str]:
    """Full pipeline: body text → dict of extracted fields."""
    token_tags = predict_tags(body, clf, vec, le)
    return extract_entities(token_tags)


# ---------------------------------------------------------------------------
# Email header parsing (minimal, for full-email input)
# ---------------------------------------------------------------------------

import re

_TO_RE = re.compile(r"^To:\s*(.+)", re.MULTILINE)
_FROM_RE = re.compile(r"^From:\s*(.+)", re.MULTILINE)
_SUBJECT_RE = re.compile(r"^Subject:\s*(.+)", re.MULTILINE)


def parse_email_text(text: str) -> dict[str, str]:
    """Parse a full email (with headers) and return header fields + body."""
    to_m = _TO_RE.search(text)
    from_m = _FROM_RE.search(text)
    subj_m = _SUBJECT_RE.search(text)

    to_addr = to_m.group(1).strip() if to_m else ""
    from_addr = from_m.group(1).strip() if from_m else ""
    subject = subj_m.group(1).strip() if subj_m else ""

    body = text
    if subj_m:
        body = text[subj_m.end():].strip()

    return {"to": to_addr, "from": from_addr, "subject": subject, "body": body}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

DISPLAY_FIELDS = [
    "serial_number",
    "item_name",
    "item_description",
    "item_specification",
    "item_quantity",
    "item_deadline",
]


def display_table(rows: list[dict], output_path: str | None = None):
    """Display rows in a pretty table and optionally save to CSV."""
    df = pd.DataFrame(rows, columns=DISPLAY_FIELDS)
    # Truncate long columns for display
    display_df = df.copy()
    for col in ["item_description", "item_specification"]:
        display_df[col] = display_df[col].apply(
            lambda x: (x[:60] + "...") if isinstance(x, str) and len(x) > 63 else x
        )

    print("\n" + "=" * 120)
    print("  EXTRACTED STRUCTURED DATA FROM EMAILS")
    print("=" * 120)
    print(
        tabulate(
            display_df,
            headers="keys",
            tablefmt="fancy_grid",
            showindex=True,
            maxcolwidths=65,
        )
    )
    print(f"\nTotal records: {len(rows)}")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Extract fields from emails using trained NER model")
    parser.add_argument("--file", type=str, help="Path to a text file containing one email")
    parser.add_argument("--stdin", action="store_true", help="Read a single email from stdin")
    parser.add_argument("--output", type=str, default=None, help="Save extracted results to this CSV file")
    parser.add_argument("--compare", action="store_true", help="Compare predictions against ground truth (uses output.csv)")
    args = parser.parse_args()

    print("Loading model...")
    clf, vec, le = load_model()

    if args.stdin:
        # Single email from stdin
        text = sys.stdin.read()
        parsed = parse_email_text(text)
        result = predict_email(parsed["body"], clf, vec, le)
        rows = [result]
        display_table(rows, args.output)
        return

    if args.file:
        # Single email from file
        text = Path(args.file).read_text(encoding="utf-8")
        parsed = parse_email_text(text)
        result = predict_email(parsed["body"], clf, vec, le)
        rows = [result]
        display_table(rows, args.output)
        return

    # Default: run on the full dataset (output.csv)
    print("Running predictions on output.csv...")
    records = load_dataset(CSV_PATH)
    rows = []

    for rec in records:
        body = rec["body"]
        result = predict_email(body, clf, vec, le)
        rows.append(result)

    display_table(rows, args.output)

    if args.compare:
        print("\n" + "=" * 120)
        print("  COMPARISON: Predicted vs Ground Truth")
        print("=" * 120)
        correct = {f: 0 for f in DISPLAY_FIELDS}
        total = len(records)

        for i, (rec, pred) in enumerate(zip(records, rows)):
            for field in DISPLAY_FIELDS:
                gt = rec.get(field, "").strip()
                pr = pred.get(field, "").strip()
                if gt.lower() == pr.lower():
                    correct[field] += 1

        print(f"\n{'Field':<25s} {'Accuracy':>10s}")
        print("-" * 37)
        for field in DISPLAY_FIELDS:
            acc = correct[field] / total * 100
            print(f"{field:<25s} {acc:9.1f}%")
        overall = sum(correct.values()) / (total * len(DISPLAY_FIELDS)) * 100
        print("-" * 37)
        print(f"{'Overall':<25s} {overall:9.1f}%")


if __name__ == "__main__":
    main()
