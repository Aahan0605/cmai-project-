-- Schema v1 -> v2: record library versions per run (TRD 5, 16 reproducibility).
-- JSON object of package -> version, e.g. {"python": "3.11.14", "torch": "2.14.1", ...}.
-- NULL for runs written before this migration. init_schema sets user_version = 2.
ALTER TABLE runs ADD COLUMN library_versions_json TEXT;
