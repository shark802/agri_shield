# No-API Automation: Salesforce Excel -> Workday Import CSV

If you do **not** have Salesforce API or Workday API access, you can still automate most of the process:

1. Export Salesforce Cases to Excel/CSV (manually or scheduled report email).
2. Run a local script to convert those rows to a Workday import CSV.
3. Upload that CSV in Workday (manual step or desktop RPA automation).

This flow uses:
- `salesforce_excel_to_workday_import.py`
- `.env.salesforce_excel_workday.example`

---

## 1) Configure

```bash
cp .env.salesforce_excel_workday.example .env
```

Then edit `.env`:

- Set column names to match your Salesforce export headers (`SF_COL_*`).
- Set `SF_WORKED_BY_VALUES` to your name/email from the export.
- Set handled statuses in `SF_INCLUDED_STATUSES`.
- Set Workday constants:
  - `WORKDAY_IMPORT_WORKER_ID`
  - `WORKDAY_IMPORT_TIME_TYPE_CODE`
  - optional project/task codes

---

## 2) Run conversion

Dry-run first:

```bash
python3 salesforce_excel_to_workday_import.py --input sf_cases.xlsx --dry-run
```

Generate output file:

```bash
python3 salesforce_excel_to_workday_import.py --input sf_cases.xlsx --output workday_time_import.csv
```

The script:
- filters only your handled cases,
- calculates hours from `SF_COL_HOURS` (or default minutes),
- deduplicates using `.salesforce_excel_workday_state.json` so repeats are skipped.

---

## 3) Load into Workday

Use your tenant’s spreadsheet import task (names vary by tenant), for example:
- "Load Time"
- "Import Time Entry"
- "Enter Time by Spreadsheet"

If Workday requires a different column schema, adjust the output mapping in:
- `salesforce_excel_to_workday_import.py` (`output_headers` and `output_row`)

---

## 4) Make it automatic (without APIs)

### Option A: Scheduled conversion + manual upload

Schedule the converter every 15 minutes (cron/Task Scheduler), then upload once daily.

### Option B: Fully automated UI upload (RPA)

Use **Power Automate Desktop** or **UiPath** to:
1. Open Workday.
2. Sign in.
3. Go to the time import task.
4. Upload the generated CSV.
5. Submit.

This gives full automation without API access.

> Keep in mind MFA/login challenges can interrupt UI bots, so add retries + alerting.

---

## Notes

- `.xlsx` input requires `openpyxl`.  
  Install with:
  ```bash
  pip install openpyxl
  ```
- `.csv` input works with no extra dependency.
- If your export has a custom hours field, set `SF_COL_HOURS` for better accuracy.
