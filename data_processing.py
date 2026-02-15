import re
import csv
from pathlib import Path

INPUT_FILE = "data.txt"
OUTPUT_FILE = "output.csv"

# Regex patterns for header fields
TO_RE = re.compile(r"^To:\s*(.+)", re.MULTILINE)
FROM_RE = re.compile(r"^From:\s*(.+)", re.MULTILINE)
SUBJECT_RE = re.compile(r"^Subject:\s*(.+)", re.MULTILINE)

# Regex patterns for embedded details in body
SERIAL_RE = re.compile(r"Serial Number\s+([A-Z0-9\-]+)")
ITEM_NAME_RE = re.compile(r'item named\s+"([^"]+)"', re.IGNORECASE)
DESC_RE = re.compile(r"item description is:\s*(.+?)\s*(?:\n|$)", re.IGNORECASE)
SPEC_RE = re.compile(r"item specification is:\s*(.+?)\s*(?:\n|$)", re.IGNORECASE)
QTY_RE = re.compile(r"item quantity is\s+(\d+)\s+units", re.IGNORECASE)
# Also allow "The required item quantity is 150 units" etc.
QTY_FALLBACK_RE = re.compile(r"item quantity\s+is\s+(\d+)\s+units", re.IGNORECASE)
QTY_GENERAL_RE = re.compile(r"item quantity\s+(\d+)\s+units", re.IGNORECASE)
DEADLINE_RE = re.compile(r"item deadline is\s+([0-9\-]+)", re.IGNORECASE)

def extract_first(pattern, text, default=""):
    m = pattern.search(text)
    return m.group(1).strip() if m else default

def extract_quantity(text):
    for pat in (QTY_RE, QTY_FALLBACK_RE, QTY_GENERAL_RE):
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    return ""

def parse_email_block(block: str) -> dict:
    block = block.strip()
    if not block:
        return None

    to_addr = extract_first(TO_RE, block)
    from_addr = extract_first(FROM_RE, block)
    subject = extract_first(SUBJECT_RE, block)

    # Body = everything after the first blank line following Subject
    # Find position of a blank line after Subject
    body = ""
    subject_match = SUBJECT_RE.search(block)
    if subject_match:
        start = subject_match.end()
        # From there onwards, strip leading newlines once
        body = block[start:].lstrip("\n")

    serial_number = extract_first(SERIAL_RE, body)
    item_name = extract_first(ITEM_NAME_RE, body)
    item_description = extract_first(DESC_RE, body)
    item_spec = extract_first(SPEC_RE, body)
    item_quantity = extract_quantity(body)
    item_deadline = extract_first(DEADLINE_RE, body)

    return {
        "source_email": block,
        "to": to_addr,
        "from": from_addr,
        "subject": subject,
        "body": body.strip(),
        "serial_number": serial_number,
        "item_name": item_name,
        "item_description": item_description,
        "item_specification": item_spec,
        "item_quantity": item_quantity,
        "item_deadline": item_deadline,
    }

def main():
    text = Path(INPUT_FILE).read_text(encoding="utf-8")

    # Split by EMAIL START / END markers
    raw_blocks = re.split(r"-{5}EMAIL START-{5}", text)
    records = []

    for blk in raw_blocks:
        blk = blk.strip()
        if not blk:
            continue
        # Ensure we only keep content up to EMAIL END for safety
        if "-----EMAIL END-----" in blk:
            blk = blk.split("-----EMAIL END-----", 1)[0].strip()
        rec = parse_email_block(blk)
        if rec:
            records.append(rec)

    fieldnames = [
        "source_email",
        "to",
        "from",
        "subject",
        "body",
        "serial_number",
        "item_name",
        "item_description",
        "item_specification",
        "item_quantity",
        "item_deadline",
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
