-- A jointly trained head produces one score, rather than a score for each input stream.
ALTER TABLE fusion_weights RENAME TO fusion_weights_v4;
CREATE TABLE fusion_weights (
    run_id INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    modality TEXT NOT NULL CHECK (modality IN ('visual', 'audio', 'frequency', 'joint')),
    weight REAL NOT NULL CHECK (weight >= 0 AND weight <= 1),
    contribution REAL CHECK (contribution IS NULL OR (contribution >= 0 AND contribution <= 1)),
    PRIMARY KEY (run_id, modality)
);
INSERT INTO fusion_weights SELECT run_id, modality, weight, contribution FROM fusion_weights_v4;
DROP TABLE fusion_weights_v4;
