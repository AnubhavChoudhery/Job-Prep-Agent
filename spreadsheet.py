# spreadsheet.py
from __future__ import annotations
import os, csv, datetime as dt
from pathlib import Path
from typing import List, Union, Optional

try:
    import gspread 
except Exception:
    gspread = None


def _now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class SpreadsheetAgent:
    """
    One API, two backends:
      - mode='csv': append to a local CSV file.
      - mode='gsheets': append to a Google Sheets worksheet using a Service Account.
    """
    def __init__(
        self,
        mode: str = "csv",
        csv_path: Optional[Union[str, Path]] = None,
        gsa_json_path: Optional[Union[str, Path]] = None,
        gsheet_id: Optional[str] = None,
        worksheet_name: str = "Sheet1",
        debug: bool = False,
    ):
        mode = (mode or "csv").lower()
        if mode not in {"csv", "gsheets"}:
            raise ValueError("mode must be 'csv' or 'gsheets'")
        self.mode = mode
        self.debug = bool(debug)

        # CSV backend setup
        self.csv_path = Path(csv_path or "logs/applications.csv")
        if self.mode == "csv":
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists():
                with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        ["timestamp_utc", "company", "position", "locations", "ats_score"]
                    )

        # Google Sheets backend config
        self.gsa_json_path = gsa_json_path or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        self.gsheet_id     = gsheet_id     or os.getenv("GOOGLE_SHEET_ID")
        self.worksheet     = (worksheet_name or os.getenv("GOOGLE_WORKSHEET") or "Sheet1")

        self._gs_client = None
        self._gs_sheet  = None

        if self.mode == "gsheets":
            if gspread is None:
                raise RuntimeError("gspread not installed. Run `pip install gspread google-auth` or switch to CSV mode.")
            if not (self.gsa_json_path and self.gsheet_id):
                raise RuntimeError("Missing Google Sheets config (GOOGLE_SERVICE_ACCOUNT_JSON, GOOGLE_SHEET_ID).")
            self._init_gsheets()

    def _init_gsheets(self):
        if self.debug:
            print(f"[SpreadsheetAgent] Using service account file: {self.gsa_json_path}")
            print(f"[SpreadsheetAgent] Opening sheet ID: {self.gsheet_id}; worksheet: {self.worksheet}")

        self._gs_client = gspread.service_account(filename=self.gsa_json_path)
        ss = self._gs_client.open_by_key(self.gsheet_id)
        try:
            self._gs_sheet = ss.worksheet(self.worksheet)
        except gspread.exceptions.WorksheetNotFound:
            # Create worksheet and write header row
            self._gs_sheet = ss.add_worksheet(title=self.worksheet, rows=1000, cols=10)
            self._gs_sheet.append_row(["timestamp_utc", "company", "position", "locations", "ats_score"])

    def append(
        self,
        company: str,
        position: str,
        locations: Union[str, List[str], None],
        ats_score: Union[int, float, str],
    ):
        ts = _now_iso()
        if isinstance(locations, (list, tuple)):
            locations = " | ".join([str(x).strip() for x in locations if str(x).strip()])
        row = [
            ts,
            str(company or "").strip(),
            str(position or "").strip(),
            str(locations or "").strip(),
            str(ats_score),
        ]

        if self.debug:
            print(f"[SpreadsheetAgent] Append row ({self.mode}): {row}")

        if self.mode == "csv":
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        else:
            self._gs_sheet.append_row(row)

    # Simple test helper
    def ping(self) -> bool:
        try:
            if self.mode == "csv":
                self.append("PING", "PING", "PING", "0")
                return True
            # gsheets: read title as a cheap permission check
            _ = self._gs_sheet.title
            return True
        except Exception as e:
            if self.debug:
                print("[SpreadsheetAgent] ping failed:", e)
            return False


def from_env(debug: bool = False) -> SpreadsheetAgent:
    """
    Convenience factory using environment variables:
      LOG_MODE=csv|gsheets
      CSV_LOG_PATH=logs/applications.csv
      GOOGLE_SERVICE_ACCOUNT_JSON=/abs/path/service_account.json
      GOOGLE_SHEET_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      GOOGLE_WORKSHEET=Sheet1
    """
    mode      = (os.getenv("LOG_MODE") or "csv").lower()
    csv_path  = os.getenv("CSV_LOG_PATH", "logs/applications.csv")
    gsa_path  = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    sheet_id  = os.getenv("GOOGLE_SHEET_ID")
    worksheet = os.getenv("GOOGLE_WORKSHEET", "Sheet1")
    return SpreadsheetAgent(
        mode=mode,
        csv_path=csv_path,
        gsa_json_path=gsa_path,
        gsheet_id=sheet_id,
        worksheet_name=worksheet,
        debug=debug,
    )
