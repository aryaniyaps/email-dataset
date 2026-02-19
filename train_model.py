"""
Train an NER model to extract structured fields from factory production emails.

Approach:
  - Token-level BIO tagging using scikit-learn (LogisticRegression)
  - Features: word shape, n-grams, context window, character prefixes/suffixes
  - Labels: BIO tags for 6 entity types:
      SERIAL, ITEM_NAME, ITEM_DESC, ITEM_SPEC, ITEM_QTY, ITEM_DEADLINE

Usage:
    python train_model.py          # trains the model and saves to model/
    python train_model.py --eval   # trains, evaluates on held-out set, and saves
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENTITY_FIELDS = [
    ("serial_number", "SERIAL"),
    ("item_name", "ITEM_NAME"),
    ("item_description", "ITEM_DESC"),
    ("item_specification", "ITEM_SPEC"),
    ("item_quantity", "ITEM_QTY"),
    ("item_deadline", "ITEM_DEADLINE"),
]

MODEL_DIR = Path("model")
CSV_PATH = Path("output.csv")

# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

# Tokenise into word-like chunks, splitting leading/trailing punctuation
_TOKEN_RE = re.compile(
    r'["\(\[]*'       # optional leading punctuation
    r'\S+?'             # core token (non-greedy)
    r'(?=["\)\].,;:!?]*(?:\s|$))'  # stop before trailing punct
    r'|["\(\)\[\].,;:!?]',        # or standalone punctuation
    re.VERBOSE,
)

# Simpler, robust approach: split on whitespace then sub-split leading/trailing punctuation
_PUNCT_CHARS = set('"\'\'()[].,;:!?')


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """Return list of (token, start_char, end_char) tuples.

    Splits on whitespace and further separates leading/trailing punctuation
    so that BIO boundaries align with entity values.
    """
    raw_tokens = [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', text)]
    result: list[tuple[str, int, int]] = []
    for tok, start, end in raw_tokens:
        # Peel leading punctuation
        while tok and tok[0] in _PUNCT_CHARS:
            result.append((tok[0], start, start + 1))
            start += 1
            tok = tok[1:]
        # Peel trailing punctuation
        trailing: list[tuple[str, int, int]] = []
        while tok and tok[-1] in _PUNCT_CHARS:
            end -= 1
            trailing.append((tok[-1], end, end + 1))
            tok = tok[:-1]
        if tok:
            result.append((tok, start, end))
        result.extend(reversed(trailing))
    return result


# ---------------------------------------------------------------------------
# BIO tag assignment
# ---------------------------------------------------------------------------


def _find_entity_span(body: str, value: str, field_key: str = "") -> tuple[int, int] | None:
    """Find the character offsets of *value* inside *body*.

    Uses context-aware matching: for short values that could appear multiple
    times (e.g. digits), search near the expected anchor phrase.
    """
    if not value:
        return None

    body_lower = body.lower()
    value_lower = value.lower()

    # Anchor phrases to disambiguate short/ambiguous values
    ANCHORS = {
        "serial_number": ["serial number"],
        "item_name": ["item named"],
        "item_description": ["item description is"],
        "item_specification": ["item specification"],
        "item_quantity": ["item quantity"],
        "item_deadline": ["item deadline"],
    }

    anchors = ANCHORS.get(field_key, [])

    # Try finding value near an anchor first (within 80 chars after anchor)
    for anchor in anchors:
        anchor_idx = body_lower.find(anchor)
        if anchor_idx == -1:
            continue
        search_start = anchor_idx
        search_region = body_lower[search_start : search_start + len(anchor) + 200]
        idx = search_region.find(value_lower)
        if idx != -1:
            abs_start = search_start + idx
            return (abs_start, abs_start + len(value))

    # Fallback: plain search
    idx = body_lower.find(value_lower)
    if idx != -1:
        return (idx, idx + len(value))

    # Try with quotes stripped
    stripped = value.strip('"').strip("'")
    idx = body_lower.find(stripped.lower())
    if idx != -1:
        return (idx, idx + len(stripped))

    return None


def assign_bio_tags(
    tokens: list[tuple[str, int, int]],
    body: str,
    fields: dict[str, str],
) -> list[str]:
    """Assign BIO tags to each token based on known entity values."""
    tags = ["O"] * len(tokens)

    # Collect entity spans  (entity_label, start, end)
    entity_spans: list[tuple[str, int, int]] = []
    for field_key, label in ENTITY_FIELDS:
        value = fields.get(field_key, "")
        if not value:
            continue
        span = _find_entity_span(body, value, field_key)
        if span is None:
            continue
        entity_spans.append((label, span[0], span[1]))

    # Sort by start offset (longer spans later break ties)
    entity_spans.sort(key=lambda x: (x[1], -(x[2] - x[1])))

    for label, ent_start, ent_end in entity_spans:
        first = True
        for i, (tok, ts, te) in enumerate(tokens):
            # Token overlaps with entity span
            if te > ent_start and ts < ent_end:
                if tags[i] == "O":  # don't overwrite
                    tags[i] = f"B-{label}" if first else f"I-{label}"
                    first = False

    return tags


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def word_shape(w: str) -> str:
    """Map word to a shape string (e.g., Xxxxx, dd-dd-dd, XXXX)."""
    s = []
    for ch in w:
        if ch.isupper():
            s.append("X")
        elif ch.islower():
            s.append("x")
        elif ch.isdigit():
            s.append("d")
        else:
            s.append(ch)
    return "".join(s)


def token_features(tokens: list[tuple[str, int, int]], idx: int) -> dict:
    """Extract features for a single token at position *idx*."""
    word = tokens[idx][0]
    feats: dict[str, str | bool | float] = {
        "bias": True,
        "word.lower": word.lower(),
        "word.shape": word_shape(word),
        "word.prefix2": word[:2].lower(),
        "word.prefix3": word[:3].lower(),
        "word.suffix2": word[-2:].lower(),
        "word.suffix3": word[-3:].lower(),
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.has_hyphen": "-" in word,
        "word.has_equals": "=" in word,
        "word.has_semicolon": ";" in word,
        "word.has_at": "@" in word,
        "word.len": len(word),
    }

    # Context window of ±3 tokens
    for offset in [-3, -2, -1, 1, 2, 3]:
        j = idx + offset
        if 0 <= j < len(tokens):
            ctx = tokens[j][0]
            prefix = f"{offset:+d}"
            feats[f"{prefix}:word.lower"] = ctx.lower()
            feats[f"{prefix}:word.shape"] = word_shape(ctx)
            feats[f"{prefix}:word.istitle"] = ctx.istitle()
            feats[f"{prefix}:word.isdigit"] = ctx.isdigit()
        else:
            feats[f"{offset:+d}:BOS_EOS"] = True

    # Position in document (normalised)
    feats["position"] = idx / max(len(tokens) - 1, 1)

    return feats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(csv_path: Path) -> list[dict]:
    """Load output.csv and return list of record dicts."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_training_data(records: list[dict]):
    """
    Convert records into (X_features, y_tags) for all tokens across all emails.
    Also returns the per-email token/tag lists for evaluation.
    """
    all_features: list[dict] = []
    all_tags: list[str] = []
    email_data: list[dict] = []  # for per-email evaluation later

    for rec in records:
        body = rec["body"]
        if not body.strip():
            continue

        tokens = tokenize(body)
        tags = assign_bio_tags(tokens, body, rec)

        feats = [token_features(tokens, i) for i in range(len(tokens))]
        all_features.extend(feats)
        all_tags.extend(tags)

        email_data.append(
            {
                "body": body,
                "tokens": tokens,
                "tags": tags,
                "fields": {k: rec[k] for k, _ in ENTITY_FIELDS},
            }
        )

    return all_features, all_tags, email_data


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train(
    features: list[dict],
    tags: list[str],
    eval_mode: bool = False,
):
    """Train a LogisticRegression token classifier. Return (model, vectorizer, label_encoder)."""
    vec = DictVectorizer(sparse=True)
    le = LabelEncoder()

    X = vec.fit_transform(features)
    y = le.fit_transform(tags)

    if eval_mode:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None

    print(f"Training on {X_train.shape[0]} tokens ({X_train.shape[1]} features)...")
    clf = LogisticRegression(
        max_iter=500,
        C=1.0,
        solver="saga",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    print(f"Training accuracy: {train_acc:.4f}")

    if eval_mode and X_test is not None:
        test_acc = clf.score(X_test, y_test)
        print(f"Test accuracy:     {test_acc:.4f}\n")

        y_pred = clf.predict(X_test)
        target_names = le.inverse_transform(sorted(set(y_test) | set(y_pred)))
        print(
            classification_report(
                le.inverse_transform(y_test),
                le.inverse_transform(y_pred),
                labels=target_names,
                zero_division=0,
            )
        )

    return clf, vec, le


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------


def save_model(clf, vec, le, model_dir: Path = MODEL_DIR):
    model_dir.mkdir(exist_ok=True)
    with open(model_dir / "classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(model_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)
    with open(model_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print(f"\nModel saved to {model_dir}/")


def load_model(model_dir: Path = MODEL_DIR):
    with open(model_dir / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(model_dir / "vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    with open(model_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return clf, vec, le


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train email NER model")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Hold out 20%% for evaluation and print classification report",
    )
    args = parser.parse_args()

    print("Loading dataset...")
    records = load_dataset(CSV_PATH)
    print(f"  {len(records)} emails loaded.\n")

    print("Building training data (BIO-tagged tokens)...")
    features, tags, email_data = build_training_data(records)

    # Show tag distribution
    from collections import Counter

    tag_counts = Counter(tags)
    print(f"  Total tokens: {len(tags)}")
    print(f"  Tag distribution:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"    {tag:20s} {count:6d}  ({100*count/len(tags):5.1f}%)")
    print()

    print("Training model...")
    clf, vec, le = train(features, tags, eval_mode=args.eval)

    save_model(clf, vec, le)
    print("Done!")


if __name__ == "__main__":
    main()
