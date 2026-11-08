"""
InvoiceIQ – Week 1: Data Cleaning & Preprocessing
Dataset: Customer Invoices Dataset (user-provided dataset.csv)

Run:
    pip install pandas numpy
    python Week1_Data_Cleaning.py

Place dataset.csv in the same folder as this script, or change INPUT_FILE below.
"""
from pathlib import Path
import pandas as pd
import numpy as np

INPUT_FILE = Path("dataset.csv")
OUTPUT_FILE = Path("Cleaned_Invoices.csv")
CLEANING_LOG = Path("Cleaning_Log.csv")

def main():
    df = pd.read_csv(INPUT_FILE, dtype={"doc_id": "string", "invoice_id": "string"})
    original_rows = len(df)
    log = []

    # Trim whitespace from text columns.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    # Remove only fully identical rows.
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates(keep="first").copy()
    log.append(("Exact duplicate rows removed", duplicate_count,
                "Removed identical copies only; retained first occurrence."))

    # Parse dates. Numeric date columns use YYYYMMDD.
    date_cols = ["clear_date", "posting_date", "document_create_date",
                 "document_create_date.1", "due_in_date", "baseline_create_date"]
    for col in date_cols:
        if col not in df.columns:
            continue
        if col in ["document_create_date", "document_create_date.1",
                   "due_in_date", "baseline_create_date"]:
            values = df[col].astype("string").str.replace(r"\.0$", "", regex=True)
            df[col] = pd.to_datetime(values, format="%Y%m%d", errors="coerce")
        else:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Convert numeric measures/status fields.
    for col in ["buisness_year", "posting_id", "area_business",
                "total_open_amount", "isOpen"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardize code/category fields.
    for col in ["business_code", "invoice_currency", "document type", "cust_payment_terms"]:
        if col in df.columns:
            df[col] = df[col].str.upper()

    # Remove columns with no non-missing information.
    all_null = [c for c in df.columns if df[c].isna().all()]
    for col in all_null:
        log.append((f"All-null column removed: {col}", len(df),
                    "Dropped because every row was missing."))
    df = df.drop(columns=all_null)

    # Map source columns to the internship project fields.
    df["invoice_id_project"] = df["doc_id"].astype("string")
    df["customer"] = df["cust_number"].astype("string") + " - " + df["name_customer"].astype("string")
    df["invoice_date"] = df["posting_date"]
    df["due_date"] = df["due_in_date"]
    df["payment_date"] = df["clear_date"]
    df["amount"] = df["total_open_amount"]
    df["currency"] = df["invoice_currency"]
    df["country"] = pd.NA  # Country is not present; do not infer from business_code.

    # Target/analysis columns. Open invoices have no known payment date,
    # so their payment-derived values remain missing (not imputed).
    df["days_to_pay"] = (df["payment_date"] - df["invoice_date"]).dt.days
    df["payment_delay_days"] = (df["payment_date"] - df["due_date"]).dt.days
    df["paid_late_flag"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    paid = df["isOpen"].eq(0) & df["payment_date"].notna()
    df.loc[paid, "paid_late_flag"] = (df.loc[paid, "payment_delay_days"] > 0).astype("int64")
    df["open_paid_flag"] = np.where(df["isOpen"].eq(1), "Open", "Paid")

    # Record key checks without silently changing questionable business values.
    due_before_invoice = int((df["due_date"] < df["invoice_date"]).sum())
    negative_amount = int((df["amount"] < 0).sum())
    log.extend([
        ("Rows before cleaning", original_rows, "Initial input row count."),
        ("Rows after cleaning", len(df), "After exact duplicate removal."),
        ("Due date before invoice/posting date", due_before_invoice,
         "Flagged for mentor review; not auto-corrected."),
        ("Negative amount values", negative_amount, "Checked; review if non-zero."),
        ("Missing country", int(df["country"].isna().sum()),
         "Country unavailable in source; left blank."),
    ])

    df.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d")
    pd.DataFrame(log, columns=["Issue", "Count", "Action / Reason"]).to_csv(CLEANING_LOG, index=False)
    print(f"Input rows: {original_rows}")
    print(f"Output rows: {len(df)}")
    print(f"Exact duplicates removed: {duplicate_count}")
    print(f"Saved: {OUTPUT_FILE.resolve()}")
    print(f"Saved: {CLEANING_LOG.resolve()}")

if __name__ == "__main__":
    main()
