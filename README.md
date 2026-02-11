# AgriShield ML Cloud Repo

## Structure
- deployment/ - ONNX Heroku API, forecasting, training script
- colab/ - Google Colab training integration + DB migration + helper PHP
- training/ - independent training services (Heroku / Railway)
- salesforce_workday_time_sync.py - standalone Salesforce -> Workday time sync script
- SALESFORCE_WORKDAY_AUTOMATION.md - setup guide for time-entry automation
- salesforce_excel_to_workday_import.py - no-API Excel/CSV -> Workday import converter
- SALESFORCE_EXCEL_TO_WORKDAY_FILE_IMPORT.md - no-API setup guide

See subfolder READMEs for detailed setup instructions.
