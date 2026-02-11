#!/usr/bin/env python3
"""Sync Salesforce Case activity into Workday time entries.

This script is intentionally generic because Workday tenants usually expose
different inbound integration endpoints. It handles:
  1) Salesforce authentication and Case querying
  2) Case -> Workday payload mapping
  3) Idempotent state tracking to prevent duplicate submissions
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


LOGGER = logging.getLogger("salesforce_workday_sync")
DEFAULT_STATE_FILE = ".salesforce_workday_sync_state.json"
DEFAULT_TIMEOUT_SECONDS = 30


def utc_now() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def parse_datetime(value: str) -> datetime:
    """Parse Salesforce and ISO-8601 datetime formats."""
    if not value:
        raise ValueError("Empty datetime string")

    formats = (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
    )
    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def format_soql_datetime(value: datetime) -> str:
    """Format datetime for SOQL WHERE clauses."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def as_float(value: Any, field_name: str) -> float | None:
    """Safely parse numeric values from API records."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse {field_name} value '{value}' as float") from exc


def parse_csv(value: str | None) -> tuple[str, ...]:
    """Split comma-separated values into a cleaned tuple."""
    if not value:
        return tuple()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def escape_soql_literal(value: str) -> str:
    """Escape text so it can be safely wrapped in SOQL single quotes."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


@dataclass(frozen=True)
class Config:
    # Salesforce
    salesforce_login_url: str
    salesforce_client_id: str
    salesforce_client_secret: str
    salesforce_username: str
    salesforce_password: str
    salesforce_security_token: str
    salesforce_api_version: str
    salesforce_owner_id: str
    salesforce_included_statuses: tuple[str, ...]
    salesforce_case_hours_field: str | None
    salesforce_case_date_field: str

    # Workday
    workday_token_url: str
    workday_client_id: str
    workday_client_secret: str
    workday_time_entry_endpoint: str
    workday_worker_id: str
    workday_time_type_code: str
    workday_project_code: str | None
    workday_task_code: str | None
    workday_custom_headers: dict[str, str]

    # Sync behavior
    state_file: Path
    default_minutes_per_case: int
    max_state_keys: int
    request_timeout_seconds: int

    @staticmethod
    def _require_env(name: str) -> str:
        value = os.getenv(name, "").strip()
        if not value:
            raise ValueError(f"Missing required environment variable: {name}")
        return value

    @classmethod
    def load(cls, state_file: Path | None = None) -> "Config":
        load_dotenv()

        statuses = parse_csv(os.getenv("SALESFORCE_INCLUDED_STATUSES", "Closed,Resolved"))
        if not statuses:
            raise ValueError("SALESFORCE_INCLUDED_STATUSES must include at least one status.")

        custom_headers_text = os.getenv("WORKDAY_CUSTOM_HEADERS_JSON", "{}").strip()
        try:
            custom_headers = json.loads(custom_headers_text)
        except json.JSONDecodeError as exc:
            raise ValueError("WORKDAY_CUSTOM_HEADERS_JSON must be valid JSON.") from exc
        if not isinstance(custom_headers, dict):
            raise ValueError("WORKDAY_CUSTOM_HEADERS_JSON must be a JSON object.")

        resolved_state_file = state_file or Path(
            os.getenv("SYNC_STATE_FILE", DEFAULT_STATE_FILE)
        )

        return cls(
            salesforce_login_url=os.getenv("SALESFORCE_LOGIN_URL", "https://login.salesforce.com"),
            salesforce_client_id=cls._require_env("SALESFORCE_CLIENT_ID"),
            salesforce_client_secret=cls._require_env("SALESFORCE_CLIENT_SECRET"),
            salesforce_username=cls._require_env("SALESFORCE_USERNAME"),
            salesforce_password=cls._require_env("SALESFORCE_PASSWORD"),
            salesforce_security_token=cls._require_env("SALESFORCE_SECURITY_TOKEN"),
            salesforce_api_version=os.getenv("SALESFORCE_API_VERSION", "v61.0"),
            salesforce_owner_id=cls._require_env("SALESFORCE_OWNER_ID"),
            salesforce_included_statuses=statuses,
            salesforce_case_hours_field=os.getenv("SALESFORCE_CASE_HOURS_FIELD", "").strip() or None,
            salesforce_case_date_field=os.getenv("SALESFORCE_CASE_DATE_FIELD", "ClosedDate"),
            workday_token_url=cls._require_env("WORKDAY_TOKEN_URL"),
            workday_client_id=cls._require_env("WORKDAY_CLIENT_ID"),
            workday_client_secret=cls._require_env("WORKDAY_CLIENT_SECRET"),
            workday_time_entry_endpoint=cls._require_env("WORKDAY_TIME_ENTRY_ENDPOINT"),
            workday_worker_id=cls._require_env("WORKDAY_WORKER_ID"),
            workday_time_type_code=cls._require_env("WORKDAY_TIME_TYPE_CODE"),
            workday_project_code=os.getenv("WORKDAY_PROJECT_CODE", "").strip() or None,
            workday_task_code=os.getenv("WORKDAY_TASK_CODE", "").strip() or None,
            workday_custom_headers={str(k): str(v) for k, v in custom_headers.items()},
            state_file=resolved_state_file,
            default_minutes_per_case=int(os.getenv("DEFAULT_MINUTES_PER_CASE", "15")),
            max_state_keys=int(os.getenv("MAX_STATE_KEYS", "10000")),
            request_timeout_seconds=int(
                os.getenv("REQUEST_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
            ),
        )


@dataclass
class SyncState:
    last_cursor_utc: datetime
    processed_keys: list[str]

    @classmethod
    def load_or_create(cls, path: Path, lookback_hours: int) -> "SyncState":
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                last_cursor_utc=parse_datetime(raw["last_cursor_utc"]),
                processed_keys=list(raw.get("processed_keys", [])),
            )

        return cls(
            last_cursor_utc=utc_now() - timedelta(hours=lookback_hours),
            processed_keys=[],
        )

    def save(self, path: Path, max_state_keys: int) -> None:
        trimmed = self.processed_keys[-max_state_keys:]
        payload = {
            "last_cursor_utc": self.last_cursor_utc.astimezone(timezone.utc).isoformat(),
            "processed_keys": trimmed,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class SalesforceClient:
    """Minimal Salesforce REST API client for Case querying."""

    def __init__(self, config: Config):
        self._config = config
        self._session = requests.Session()
        self._instance_url = ""

    def authenticate(self) -> None:
        token_url = f"{self._config.salesforce_login_url}/services/oauth2/token"
        body = {
            "grant_type": "password",
            "client_id": self._config.salesforce_client_id,
            "client_secret": self._config.salesforce_client_secret,
            "username": self._config.salesforce_username,
            "password": f"{self._config.salesforce_password}{self._config.salesforce_security_token}",
        }
        response = self._session.post(
            token_url,
            data=body,
            timeout=self._config.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        access_token = payload["access_token"]
        self._instance_url = payload["instance_url"]
        self._session.headers.update({"Authorization": f"Bearer {access_token}"})
        LOGGER.info("Authenticated with Salesforce org at %s", self._instance_url)

    def query_handled_cases(self, updated_after_utc: datetime, max_cases: int) -> list[dict[str, Any]]:
        if not self._instance_url:
            raise RuntimeError("Salesforce client is not authenticated.")

        case_fields = [
            "Id",
            "CaseNumber",
            "Subject",
            "Status",
            "OwnerId",
            "LastModifiedDate",
            "ClosedDate",
        ]

        if self._config.salesforce_case_date_field not in case_fields:
            case_fields.append(self._config.salesforce_case_date_field)
        if self._config.salesforce_case_hours_field:
            case_fields.append(self._config.salesforce_case_hours_field)

        escaped_statuses = ", ".join(
            f"'{escape_soql_literal(status)}'"
            for status in self._config.salesforce_included_statuses
        )
        soql = (
            f"SELECT {', '.join(case_fields)} "
            f"FROM Case "
            f"WHERE OwnerId = '{self._config.salesforce_owner_id}' "
            f"AND Status IN ({escaped_statuses}) "
            f"AND LastModifiedDate >= {format_soql_datetime(updated_after_utc)} "
            f"ORDER BY LastModifiedDate ASC "
            f"LIMIT {max_cases}"
        )

        url = f"{self._instance_url}/services/data/{self._config.salesforce_api_version}/query"
        response = self._session.get(
            url,
            params={"q": soql},
            timeout=self._config.request_timeout_seconds,
        )
        response.raise_for_status()
        records = response.json().get("records", [])
        LOGGER.info("Fetched %d candidate Cases from Salesforce", len(records))
        return records


class WorkdayClient:
    """Client for posting time entries to a Workday integration endpoint."""

    def __init__(self, config: Config):
        self._config = config
        self._session = requests.Session()
        self._access_token: str | None = None

    def authenticate(self) -> None:
        response = self._session.post(
            self._config.workday_token_url,
            data={"grant_type": "client_credentials"},
            auth=(self._config.workday_client_id, self._config.workday_client_secret),
            timeout=self._config.request_timeout_seconds,
        )
        response.raise_for_status()
        self._access_token = response.json()["access_token"]
        LOGGER.info("Obtained Workday OAuth token")

    def submit_time_entry(self, payload: dict[str, Any]) -> None:
        if not self._access_token:
            raise RuntimeError("Workday client is not authenticated.")

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            **self._config.workday_custom_headers,
        }
        response = self._session.post(
            self._config.workday_time_entry_endpoint,
            json=payload,
            headers=headers,
            timeout=self._config.request_timeout_seconds,
        )

        # Duplicate payloads may be rejected by design if your endpoint enforces idempotency.
        if response.status_code == 409:
            LOGGER.info(
                "Workday endpoint reported duplicate entry for reference=%s",
                payload.get("externalReference"),
            )
            return

        response.raise_for_status()


def build_workday_payload(config: Config, case_record: dict[str, Any]) -> dict[str, Any]:
    """Map one Salesforce Case record to one Workday time payload."""
    hours = config.default_minutes_per_case / 60.0
    if config.salesforce_case_hours_field:
        parsed_hours = as_float(
            case_record.get(config.salesforce_case_hours_field),
            config.salesforce_case_hours_field,
        )
        if parsed_hours is not None:
            hours = parsed_hours

    workday_date_raw = (
        case_record.get(config.salesforce_case_date_field)
        or case_record.get("ClosedDate")
        or case_record["LastModifiedDate"]
    )
    workday_date = parse_datetime(workday_date_raw).date().isoformat()

    subject = (case_record.get("Subject") or "").replace("\n", " ").strip()
    comment = f"Salesforce Case {case_record.get('CaseNumber')}: {subject}".strip()

    payload: dict[str, Any] = {
        "workerId": config.workday_worker_id,
        "date": workday_date,
        "hours": round(hours, 2),
        "timeTypeCode": config.workday_time_type_code,
        "projectCode": config.workday_project_code,
        "taskCode": config.workday_task_code,
        "comment": comment[:250],
        "externalReference": f"salesforce-case-{case_record['Id']}-{case_record['LastModifiedDate']}",
        "sourceSystem": "Salesforce",
        "sourceCaseId": case_record["Id"],
        "sourceCaseNumber": case_record.get("CaseNumber"),
        "sourceCaseStatus": case_record.get("Status"),
        "sourceCaseOwnerId": case_record.get("OwnerId"),
        "sourceLastModifiedDate": case_record.get("LastModifiedDate"),
    }

    # Remove null/empty fields so tenant-specific validators do not reject payloads.
    return {key: value for key, value in payload.items() if value not in ("", None)}


def run_sync(
    config: Config,
    dry_run: bool,
    lookback_hours: int,
    max_cases: int,
    reset_state: bool,
    force_cursor_utc: datetime | None,
) -> dict[str, int]:
    """Execute one sync loop and return summary counts."""
    state_path = config.state_file
    if reset_state and state_path.exists():
        state_path.unlink()
        LOGGER.info("Deleted existing state file at %s", state_path)

    state = SyncState.load_or_create(state_path, lookback_hours)
    if force_cursor_utc:
        state.last_cursor_utc = force_cursor_utc
        LOGGER.info("Forced sync cursor to %s", state.last_cursor_utc.isoformat())

    sf_client = SalesforceClient(config)
    sf_client.authenticate()

    wd_client = WorkdayClient(config)
    if not dry_run:
        wd_client.authenticate()

    case_records = sf_client.query_handled_cases(state.last_cursor_utc, max_cases=max_cases)
    processed_key_set = set(state.processed_keys)

    created = 0
    skipped_duplicate = 0
    failed = 0
    newest_cursor = state.last_cursor_utc

    for case_record in case_records:
        external_key = f"{case_record['Id']}::{case_record['LastModifiedDate']}"
        last_modified = parse_datetime(case_record["LastModifiedDate"])
        if last_modified > newest_cursor:
            newest_cursor = last_modified

        if external_key in processed_key_set:
            skipped_duplicate += 1
            continue

        try:
            payload = build_workday_payload(config, case_record)
            if dry_run:
                LOGGER.info("DRY RUN: Would submit payload: %s", json.dumps(payload))
            else:
                wd_client.submit_time_entry(payload)
            created += 1
            state.processed_keys.append(external_key)
            processed_key_set.add(external_key)
        except Exception:  # noqa: BLE001 - keep loop running per case
            failed += 1
            LOGGER.exception(
                "Failed syncing Case id=%s caseNumber=%s",
                case_record.get("Id"),
                case_record.get("CaseNumber"),
            )

    state.last_cursor_utc = newest_cursor
    state.save(state_path, config.max_state_keys)

    return {
        "created": created,
        "skipped_duplicate": skipped_duplicate,
        "failed": failed,
        "examined": len(case_records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync Salesforce Cases to Workday time entries."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and log payloads without creating Workday time entries.",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=12,
        help="First-run lookback window when no local state file exists.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=200,
        help="Maximum Salesforce Case records to process in one run.",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Delete local sync state and start from a new lookback cursor.",
    )
    parser.add_argument(
        "--force-cursor-utc",
        type=str,
        default="",
        help="Override cursor with an explicit UTC datetime (ISO-8601).",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="",
        help=f"Optional override for state file path (default: {DEFAULT_STATE_FILE}).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    try:
        state_path = Path(args.state_file).expanduser() if args.state_file else None
        config = Config.load(state_file=state_path)
        forced_cursor = parse_datetime(args.force_cursor_utc) if args.force_cursor_utc else None

        summary = run_sync(
            config=config,
            dry_run=args.dry_run,
            lookback_hours=args.lookback_hours,
            max_cases=args.max_cases,
            reset_state=args.reset_state,
            force_cursor_utc=forced_cursor,
        )
        LOGGER.info(
            "Sync complete. examined=%d created=%d skipped_duplicate=%d failed=%d",
            summary["examined"],
            summary["created"],
            summary["skipped_duplicate"],
            summary["failed"],
        )
        return 0 if summary["failed"] == 0 else 2
    except Exception:  # noqa: BLE001
        LOGGER.exception("Salesforce -> Workday sync failed before completion")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
