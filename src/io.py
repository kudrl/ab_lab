from __future__ import annotations



from dataclasses import dataclass
from typing import Any, Dict, Optional, Set
import io
import json
import zipfile

import pandas as pd

REQUIRED_COLS: Set[str] = {"user_id", "variant", "ts", "event"}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    df: Optional[pd.DataFrame]
    error: Optional[str]


def load_and_validate_csv(file) -> ValidationResult:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return ValidationResult(False, None, f"Failed to read CSV: {e}")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return ValidationResult(False, None, f"Missing columns: {sorted(missing)}")

    df = df.copy()

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    if df["ts"].isna().any():
        return ValidationResult(False, None, "Column 'ts' has invalid datetime values")

    df["variant"] = df["variant"].astype(str).str.upper().str.strip()
    if not set(df["variant"].unique()).issubset({"A", "B"}):
        return ValidationResult(False, None, "Column 'variant' must contain only 'A'/'B'")

    if "amount" not in df.columns:
        df["amount"] = 0.0
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    df["event"] = df["event"].astype(str).str.strip()
    df["date"] = df["ts"].dt.floor("D")

    return ValidationResult(True, df, None)

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def obj_to_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def make_zip_bytes(files: Dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()
