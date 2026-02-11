# Salesforce -> Workday Time Automation

This guide shows how to auto-create Workday time entries from Salesforce Case activity so you do not need manual end-of-day entry.

## What this automation does

`salesforce_workday_time_sync.py`:

1. Authenticates to Salesforce.
2. Pulls recently handled Cases for your Salesforce owner ID.
3. Converts each Case into one Workday time-entry payload.
4. Posts payloads to your Workday integration endpoint.
5. Stores a local state file to avoid duplicate entries.

---

## 1) Prerequisites

- Python 3.10+.
- Salesforce Connected App (OAuth enabled).
- Workday API app/integration endpoint that can create time entries.
- Credentials available as environment variables.

> Workday tenants differ; many teams expose a custom endpoint (Workday Studio, Extend, or API Gateway) for time-entry ingestion.  
> This script assumes that endpoint accepts JSON payloads.

---

## 2) Configure environment variables

1. Copy the sample file:

```bash
cp .env.salesforce_workday.example .env
```

2. Fill in your actual values in `.env`.

Important fields:

- `SALESFORCE_OWNER_ID`: your Salesforce user ID (`005...`).
- `SALESFORCE_INCLUDED_STATUSES`: statuses that mean "handled" (for example `Closed,Resolved`).
- `SALESFORCE_CASE_HOURS_FIELD` (optional): if your Case has a custom hours field, set it here.
- `DEFAULT_MINUTES_PER_CASE`: fallback effort when no hours field exists.
- `WORKDAY_TIME_ENTRY_ENDPOINT`: endpoint that creates the Workday time entry.
- `WORKDAY_WORKER_ID`: your Workday worker ID.

---

## 3) Dry run first

Run in dry-run mode to validate mapping and filtering without writing to Workday:

```bash
python salesforce_workday_time_sync.py --dry-run --lookback-hours 24
```

If output looks correct, run live:

```bash
python salesforce_workday_time_sync.py --lookback-hours 24
```

---

## 4) Schedule automatic runs

Use cron every 10 minutes:

```bash
*/10 * * * * cd /path/to/repo && /usr/bin/env python salesforce_workday_time_sync.py >> sync.log 2>&1
```

The script keeps a state file (default: `.salesforce_workday_sync_state.json`) to prevent double-submission.

---

## Payload mapping (default)

Salesforce Case -> Workday payload:

- `workerId` <- `WORKDAY_WORKER_ID`
- `date` <- `SALESFORCE_CASE_DATE_FIELD` (fallback to `ClosedDate`, then `LastModifiedDate`)
- `hours` <- `SALESFORCE_CASE_HOURS_FIELD` (fallback to `DEFAULT_MINUTES_PER_CASE / 60`)
- `timeTypeCode` <- `WORKDAY_TIME_TYPE_CODE`
- `comment` <- `Salesforce Case <CaseNumber>: <Subject>`
- `externalReference` <- `salesforce-case-<CaseId>-<LastModifiedDate>`

Adjust the `build_workday_payload()` function if your Workday endpoint expects different field names.

---

## Common setup recommendations

1. **Use a Salesforce custom field for effort**  
   Add `Workday_Hours__c` (Number) and populate it through Flow or manually.

2. **Use a strict handled-status filter**  
   Start with `Closed` (or your exact final status) to avoid logging work in progress.

3. **Enforce idempotency in Workday endpoint**  
   Use `externalReference` as a unique key to reject duplicates cleanly.

4. **Start with a short lookback window**  
   Use `--lookback-hours 2` for first live run, then rely on persisted state.

---

## Troubleshooting

- `Missing required environment variable`  
  Confirm `.env` values and variable names.

- Salesforce auth errors (`invalid_grant`)  
  Verify username/password/security token and Connected App policy.

- Workday 4xx errors  
  Check endpoint URL, OAuth client, expected JSON schema, and any required headers in `WORKDAY_CUSTOM_HEADERS_JSON`.

- Duplicate entries  
  Ensure your endpoint treats `externalReference` as unique.
