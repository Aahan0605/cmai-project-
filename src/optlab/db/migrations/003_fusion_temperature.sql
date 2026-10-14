-- Schema v2 -> v3: calibrated fusion (Gate 1 finding, docs/baseline_analysis.md section 5).
-- Plain simplex fusion forces sum(w) = 1, so weights also absorb logit calibration. The
-- 'simplex_temperature' formulation fits u >= 0 (u = s w, w on the simplex, s > 0 a learned
-- temperature), which separates scale from mixing.
ALTER TABLE fusion_solves ADD COLUMN formulation TEXT NOT NULL DEFAULT 'simplex'
    CHECK (formulation IN ('simplex', 'simplex_temperature'));
ALTER TABLE fusion_solves ADD COLUMN temperature REAL;          -- s = sum(u); NULL for 'simplex'
-- Share of the fused logit's spread each stream contributes on val_fusion:
-- w_m * sd(z_m) / sum_k w_k * sd(z_k). Unlike w itself, this does not depend on logit scale.
ALTER TABLE fusion_weights ADD COLUMN contribution REAL CHECK (contribution IS NULL OR (contribution >= 0 AND contribution <= 1));
