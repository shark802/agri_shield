#!/usr/bin/env python3
"""Build Workday time-import CSV from exported Salesforce Case files.

Use this when direct Salesforce/Workday APIs are unavailable.
Input formats:
  - .csv (recommended)
  - .xlsx (requires openpyxl)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

try:
    import openpyxl
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    openpyxl = None


LOGGER = logging.getLogger("salesforce_excel_to_workday_import")
DEFAULT_STATE_FILE = ".salesforce_excel_workday_state.json"


def normalize_header(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def parse_csv_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return tuple()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def parse_date_like(value: Any) -> date:
    """Parse dates from excel/csv representations into a date object."""
    if value is None or value == "":
        raise ValueError("Empty date value")

    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()

    text = str(value).strip()
    if not text:
        raise ValueError("Empty date text")

    formats = (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    # Final fallback for ISO-ish strings with timezone offset.
    return datetime.fromisoformat(text.replace("Z", "+00:00")).date()


def parse_hours(value: Any) -> float | None:
    """Parse decimal hours from text/numeric values."""
    if value is None or value == "":
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    # Handle HH:MM style values.
    if re.fullmatch(r"\d{1,2}:\d{2}", text):
        hours_part, minutes_part = text.split(":")
        return int(hours_part) + (int(minutes_part) / 60.0)

    normalized = (
        text.lower()
        .replace("hours", "")
        .replace("hour", "")
        .replace("hrs", "")
        .replace("hr", "")
        .strip()
    )
    return float(normalized)


def safe_comment_template(template: str, fields: dict[str, str]) -> str:
    """Render comment template safely, falling back to default style."""
    try:
        return template.format(**fields).strip()
    except KeyError:
        return f"Salesforce Case {fields.get('case_number', '')}: {fields.get('subject', '')}".strip()


def sanitize_reference(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._:-]+", "-", value).strip("-")


class ColumnResolver:
    """Resolve exact or fuzzy header matches from exported files."""

    def __init__(self, headers: list[str]):
        self.headers = headers
        self._normalized_to_actual: dict[str, str] = {}
        for header in headers:
            if header:
                self._normalized_to_actual[normalize_header(header)] = header

    def resolve(self, configured_name: str, required: bool = True) -> str | None:
        candidate = configured_name.strip()
        if not candidate:
            if required:
                raise ValueError("Required column name is empty.")
            return None

        if candidate in self.headers:
            return candidate

        fuzzy_key = normalize_header(candidate)
        resolved = self._normalized_to_actual.get(fuzzy_key)
        if resolved:
            return resolved

        if required:
            raise ValueError(
                f"Column '{configured_name}' not found in input. "
                f"Available headers: {', '.join(self.headers)}"
            )
        return None


@dataclass(frozen=True)
class Config:
    # Input
    sf_col_case_id: str
    sf_col_case_number: str
    sf_col_owner: str
    sf_col_status: str
    sf_col_date: str
    sf_col_subject: str
    sf_col_hours: str | None
    sf_col_last_modified: str | None
    sf_worked_by_values: tuple[str, ...]
    sf_included_statuses: tuple[str, ...]
    sf_date_from: date | None
    sf_date_to: date | None

    # Workday output constants
    workday_worker_id: str
    workday_time_type_code: str
    workday_project_code: str | None
    workday_task_code: str | None
    workday_comment_template: str
    external_reference_prefix: str

    # Behavior
    default_minutes_per_case: int
    dedupe_on_case_id_and_date: bool
    max_state_keys: int

    @staticmethod
    def _date_from_env(name: str) -> date | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        return parse_date_like(raw)

    @classmethod
    def load(cls) -> "Config":
        load_dotenv()

        worked_by = parse_csv_list(os.getenv("SF_WORKED_BY_VALUES", ""))
        statuses = parse_csv_list(os.getenv("SF_INCLUDED_STATUSES", "Closed,Resolved"))
        if not statuses:
            raise ValueError("SF_INCLUDED_STATUSES must include at least one value.")

        worker_id = os.getenv("WORKDAY_IMPORT_WORKER_ID", "").strip()
        if not worker_id:
            raise ValueError("WORKDAY_IMPORT_WORKER_ID is required.")

        time_type = os.getenv("WORKDAY_IMPORT_TIME_TYPE_CODE", "").strip()
        if not time_type:
            raise ValueError("WORKDAY_IMPORT_TIME_TYPE_CODE is required.")

        sf_col_hours = os.getenv("SF_COL_HOURS", "").strip() or None
        sf_col_last_modified = os.getenv("SF_COL_LAST_MODIFIED", "").strip() or None

        return cls(
            sf_col_case_id=os.getenv("SF_COL_CASE_ID", "Case ID").strip(),
            sf_col_case_number=os.getenv("SF_COL_CASE_NUMBER", "Case Number").strip(),
            sf_col_owner=os.getenv("SF_COL_OWNER", "Case Owner").strip(),
            sf_col_status=os.getenv("SF_COL_STATUS", "Status").strip(),
            sf_col_date=os.getenv("SF_COL_DATE", "Closed Date").strip(),
            sf_col_subject=os.getenv("SF_COL_SUBJECT", "Subject").strip(),
            sf_col_hours=sf_col_hours,
            sf_col_last_modified=sf_col_last_modified,
            sf_worked_by_values=tuple(value.lower() for value in worked_by),
            sf_included_statuses=tuple(value.lower() for value in statuses),
            sf_date_from=cls._date_from_env("SF_DATE_FROM"),
            sf_date_to=cls._date_from_env("SF_DATE_TO"),
            workday_worker_id=worker_id,
            workday_time_type_code=time_type,
            workday_project_code=os.getenv("WORKDAY_IMPORT_PROJECT_CODE", "").strip() or None,
            workday_task_code=os.getenv("WORKDAY_IMPORT_TASK_CODE", "").strip() or None,
            workday_comment_template=os.getenv(
                "WORKDAY_IMPORT_COMMENT_TEMPLATE",
                "Salesforce Case {case_number}: {subject}",
            ),
            external_reference_prefix=os.getenv(
                "EXTERNAL_REFERENCE_PREFIX", "salesforce-case"
            ).strip(),
            default_minutes_per_case=int(os.getenv("DEFAULT_MINUTES_PER_CASE", "15")),
            dedupe_on_case_id_and_date=parse_bool(
                os.getenv("DEDUPE_ON_CASE_ID_AND_DATE", "true"),
                default=True,
            ),
            max_state_keys=int(os.getenv("MAX_STATE_KEYS", "10000")),
        )


@dataclass
class State:
    processed_keys: list[str]

    @classmethod
    def load_or_create(cls, path: Path) -> "State":
        if not path.exists():
            return cls(processed_keys=[])
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(processed_keys=list(raw.get("processed_keys", [])))

    def save(self, path: Path, max_state_keys: int) -> None:
        payload = {"processed_keys": self.processed_keys[-max_state_keys:]}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_input_rows(input_file: Path) -> tuple[list[dict[str, Any]], list[str]]:
    suffix = input_file.suffix.lower()
    if suffix == ".csv":
        with input_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
            headers = list(reader.fieldnames or [])
        return rows, headers

    if suffix == ".xlsx":
        if openpyxl is None:
            raise RuntimeError(
                "openpyxl is required for .xlsx input. Install with: pip install openpyxl"
            )
        workbook = openpyxl.load_workbook(input_file, read_only=True, data_only=True)
        worksheet = workbook.active

        row_iterator = worksheet.iter_rows(values_only=True)
        header_row = next(row_iterator, None)
        if not header_row:
            return [], []

        headers = [str(cell).strip() if cell is not None else "" for cell in header_row]
        rows: list[dict[str, Any]] = []
        for values in row_iterator:
            row: dict[str, Any] = {}
            for index, header in enumerate(headers):
                row[header] = values[index] if index < len(values) else None
            rows.append(row)
        return rows, headers

    raise ValueError("Unsupported input format. Use .csv or .xlsx")


def build_dedupe_key(
    *,
    case_id: str,
    case_number: str,
    case_date: date,
    last_modified_text: str | None,
    dedupe_on_case_id_and_date: bool,
) -> str:
    if dedupe_on_case_id_and_date:
        return f"{case_id or case_number}::{case_date.isoformat()}"
    return f"{case_id or case_number}::{case_date.isoformat()}::{last_modified_text or ''}"


def convert_file(
    *,
    config: Config,
    input_file: Path,
    output_file: Path,
    state_file: Path,
    dry_run: bool,
    reset_state: bool,
) -> dict[str, int]:
    if reset_state and state_file.exists():
        state_file.unlink()
        LOGGER.info("Deleted previous state file at %s", state_file)

    state = State.load_or_create(state_file)
    dedupe_set = set(state.processed_keys)

    rows, headers = read_input_rows(input_file)
    if not rows:
        LOGGER.warning("Input file had no data rows.")
        return {"examined": 0, "written": 0, "skipped": 0, "failed": 0}

    resolver = ColumnResolver(headers)
    col_case_id = resolver.resolve(config.sf_col_case_id, required=True)
    col_case_number = resolver.resolve(config.sf_col_case_number, required=True)
    col_owner = resolver.resolve(config.sf_col_owner, required=True)
    col_status = resolver.resolve(config.sf_col_status, required=True)
    col_date = resolver.resolve(config.sf_col_date, required=True)
    col_subject = resolver.resolve(config.sf_col_subject, required=False)
    col_hours = resolver.resolve(config.sf_col_hours, required=False) if config.sf_col_hours else None
    col_last_modified = (
        resolver.resolve(config.sf_col_last_modified, required=False)
        if config.sf_col_last_modified
        else None
    )

    output_headers = [
        "Worker_ID",
        "Date",
        "Hours",
        "Time_Type_Code",
        "Project_Code",
        "Task_Code",
        "Comment",
        "External_Reference",
        "Source_Case_ID",
        "Source_Case_Number",
        "Source_Status",
        "Source_Owner",
    ]

    written_rows: list[dict[str, str]] = []
    examined = len(rows)
    written = 0
    skipped = 0
    failed = 0

    for source_row in rows:
        try:
            raw_owner = str(source_row.get(col_owner, "") or "").strip()
            raw_status = str(source_row.get(col_status, "") or "").strip()

            if config.sf_worked_by_values and raw_owner.lower() not in config.sf_worked_by_values:
                skipped += 1
                continue
            if raw_status.lower() not in config.sf_included_statuses:
                skipped += 1
                continue

            raw_case_id = str(source_row.get(col_case_id, "") or "").strip()
            raw_case_number = str(source_row.get(col_case_number, "") or "").strip()
            case_date = parse_date_like(source_row.get(col_date))
            if config.sf_date_from and case_date < config.sf_date_from:
                skipped += 1
                continue
            if config.sf_date_to and case_date > config.sf_date_to:
                skipped += 1
                continue

            raw_subject = str(source_row.get(col_subject, "") or "").replace("\n", " ").strip()
            hours = parse_hours(source_row.get(col_hours)) if col_hours else None
            if hours is None:
                hours = config.default_minutes_per_case / 60.0

            raw_last_modified = (
                str(source_row.get(col_last_modified, "") or "").strip()
                if col_last_modified
                else ""
            )
            dedupe_key = build_dedupe_key(
                case_id=raw_case_id,
                case_number=raw_case_number,
                case_date=case_date,
                last_modified_text=raw_last_modified,
                dedupe_on_case_id_and_date=config.dedupe_on_case_id_and_date,
            )
            if dedupe_key in dedupe_set:
                skipped += 1
                continue

            comment = safe_comment_template(
                config.workday_comment_template,
                {
                    "case_id": raw_case_id,
                    "case_number": raw_case_number,
                    "subject": raw_subject,
                    "status": raw_status,
                    "owner": raw_owner,
                    "date": case_date.isoformat(),
                },
            )
            external_ref = sanitize_reference(
                f"{config.external_reference_prefix}-{raw_case_id or raw_case_number}-{case_date.isoformat()}"
            )

            output_row = {
                "Worker_ID": config.workday_worker_id,
                "Date": case_date.isoformat(),
                "Hours": f"{hours:.2f}",
                "Time_Type_Code": config.workday_time_type_code,
                "Project_Code": config.workday_project_code or "",
                "Task_Code": config.workday_task_code or "",
                "Comment": comment[:250],
                "External_Reference": external_ref[:100],
                "Source_Case_ID": raw_case_id,
                "Source_Case_Number": raw_case_number,
                "Source_Status": raw_status,
                "Source_Owner": raw_owner,
            }
            written_rows.append(output_row)
            written += 1
            state.processed_keys.append(dedupe_key)
            dedupe_set.add(dedupe_key)
        except Exception:  # noqa: BLE001 - continue processing remaining rows
            failed += 1
            LOGGER.exception("Failed to process input row: %s", source_row)

    if not dry_run:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_headers)
            writer.writeheader()
            writer.writerows(written_rows)
        state.save(state_file, config.max_state_keys)
        LOGGER.info("Wrote %d Workday rows to %s", written, output_file)
    else:
        LOGGER.info("Dry run enabled; no output file written.")

    return {
        "examined": examined,
        "written": written,
        "skipped": skipped,
        "failed": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert exported Salesforce Cases (.csv/.xlsx) into Workday import CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to Salesforce export file (.csv or .xlsx).",
    )
    parser.add_argument(
        "--output",
        default="workday_time_import.csv",
        type=str,
        help="Path to output Workday CSV file.",
    )
    parser.add_argument(
        "--state-file",
        default=DEFAULT_STATE_FILE,
        type=str,
        help=f"Path for dedupe state storage (default: {DEFAULT_STATE_FILE}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and validate rows without writing output/state.",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Delete state file before processing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    try:
        config = Config.load()
        summary = convert_file(
            config=config,
            input_file=Path(args.input).expanduser(),
            output_file=Path(args.output).expanduser(),
            state_file=Path(args.state_file).expanduser(),
            dry_run=args.dry_run,
            reset_state=args.reset_state,
        )
        LOGGER.info(
            "Conversion complete. examined=%d written=%d skipped=%d failed=%d",
            summary["examined"],
            summary["written"],
            summary["skipped"],
            summary["failed"],
        )
        return 0 if summary["failed"] == 0 else 2
    except Exception:  # noqa: BLE001
        LOGGER.exception("Excel -> Workday conversion failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
