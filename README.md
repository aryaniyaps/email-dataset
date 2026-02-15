# Factory Email Extraction

This project parses a synthetic corpus of factory production emails (`data.txt`) and converts it into a structured CSV file (`output.csv`). The goal is to make it easy to run classical regex‑based information extraction experiments on realistic, natural‑language email text.

## Overview

- **Input**: `data.txt`  
  A plain‑text file containing 100 emails.  
  Each email:
  - Is wrapped between `-----EMAIL START-----` and `-----EMAIL END-----`
  - Contains `To:`, `From:`, and `Subject:` headers
  - Has a free‑form body with embedded production details such as serial number, item name, description, specification, quantity, and deadline, phrased like natural emails but using consistent anchor phrases (e.g., `Serial Number SN-0001`, `item named "..."`, `item description is: ...`).

- **Output**: `output.csv`  
  A tabular representation where each email becomes a single row and both the raw text and extracted fields are available as columns.

## Running the script

1. Make sure `data.txt` (the email corpus) is in the project root.
2. Run:

   ```bash
   python extract_emails_to_csv.py
   ```

3. The script will create `output.csv` in the same directory and print the number of rows written.

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