-- Persist the exact validation-fitted transform for reproducible evaluation/export.
ALTER TABLE fusion_solves ADD COLUMN calibration_json TEXT;
