-- PK Clearance Predictor — Database Schema
-- Based on PK-DB API field discovery + TDC ADME datasets
-- Source 1: PK-DB (pkdata/) — clinical demographics, dosing, study metadata
-- Source 2: TDC — molecular ADME endpoints (clearance, half-life, etc.)

-- =============================================================
-- SOURCE 1: PK-DB TABLES
-- =============================================================

-- Studies metadata (from pkdata/studies)
CREATE TABLE IF NOT EXISTS pkdb_studies (
    sid             TEXT PRIMARY KEY,       -- e.g. "PKDB00954"
    name            TEXT NOT NULL,          -- e.g. "Rosenkranz1996a"
    licence         TEXT,                   -- "open" or "closed"
    access          TEXT,                   -- "public"
    date            TEXT,                   -- curation date
    creator         TEXT,
    curators        TEXT,                   -- JSON array as string
    substances      TEXT,                   -- JSON array of substance names
    reference_pmid  TEXT,
    reference_title TEXT,
    reference_date  TEXT
);

-- Group-level demographics (from pkdata/groups)
-- Each row is ONE measurement for ONE group (wide-form via measurement_type)
CREATE TABLE IF NOT EXISTS pkdb_groups (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    study_sid           TEXT REFERENCES pkdb_studies(sid),
    study_name          TEXT,
    group_pk            INTEGER,
    group_name          TEXT,
    group_count         INTEGER,           -- number of subjects in group
    group_parent_pk     INTEGER,
    characteristica_pk  INTEGER,
    count               INTEGER,
    measurement_type    TEXT,               -- "age", "weight", "sex", "smoking", etc.
    calculation_type    TEXT,               -- "sample mean", "median", etc.
    choice              TEXT,               -- categorical value (e.g. "M", "homo sapiens")
    substance           TEXT,
    value               REAL,
    mean                REAL,
    median              REAL,
    min                 REAL,
    max                 REAL,
    sd                  REAL,
    se                  REAL,
    cv                  REAL,
    unit                TEXT
);

-- Individual-level demographics (from pkdata/individuals)
-- Same structure as groups but per-patient
CREATE TABLE IF NOT EXISTS pkdb_individuals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    study_sid           TEXT REFERENCES pkdb_studies(sid),
    study_name          TEXT,
    individual_pk       INTEGER,
    individual_name     TEXT,
    individual_group_pk INTEGER,
    characteristica_pk  INTEGER,
    count               INTEGER,
    measurement_type    TEXT,               -- "age", "weight", "sex", "bmi", "genotype", etc.
    calculation_type    TEXT,
    choice              TEXT,
    substance           TEXT,
    value               REAL,
    mean                REAL,
    median              REAL,
    min                 REAL,
    max                 REAL,
    sd                  REAL,
    se                  REAL,
    cv                  REAL,
    unit                TEXT
);

-- Interventions / dosing (from pkdata/interventions)
CREATE TABLE IF NOT EXISTS pkdb_interventions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    study_sid               TEXT REFERENCES pkdb_studies(sid),
    study_name              TEXT,
    intervention_pk         INTEGER,
    raw_pk                  INTEGER,
    normed                  BOOLEAN,
    name                    TEXT,
    route                   TEXT,              -- "oral", "iv", etc.
    route_label             TEXT,
    form                    TEXT,              -- "tablet", "capsule", etc.
    form_label              TEXT,
    application             TEXT,              -- "single-dose", "multiple-dose"
    application_label       TEXT,
    time                    TEXT,              -- dosing times (pipe-delimited)
    time_end                TEXT,
    time_unit               TEXT,              -- "hr", "d", "min"
    measurement_type        TEXT,
    measurement_type_label  TEXT,
    calculation_type        TEXT,
    calculation_type_label  TEXT,
    choice                  TEXT,
    choice_label            TEXT,
    substance               TEXT,              -- drug name
    substance_label         TEXT,
    value                   REAL,              -- dose value
    mean                    REAL,
    median                  REAL,
    min                     REAL,
    max                     REAL,
    sd                      REAL,
    se                      REAL,
    cv                      REAL,
    unit                    TEXT               -- "mg", "mg/kg", etc.
);

-- Substance statistics (from statistics/substances)
CREATE TABLE IF NOT EXISTS pkdb_substance_stats (
    info_node_label     TEXT PRIMARY KEY,
    output_count        INTEGER,
    intervention_count  INTEGER
);

-- Caffeine study details (from studies/<SID>/)
CREATE TABLE IF NOT EXISTS pkdb_caffeine_studies (
    sid                 TEXT PRIMARY KEY,
    name                TEXT,
    output_count        INTEGER,
    individual_count    INTEGER,
    group_count         INTEGER,
    intervention_count  INTEGER,
    timecourse_count    INTEGER,
    output_pks          TEXT,               -- JSON array of output PKs
    licence             TEXT
);

-- =============================================================
-- SOURCE 2: TDC ADME TABLES
-- =============================================================

-- Drug clearance prediction (AstraZeneca dataset)
-- PRIMARY ML TARGET
CREATE TABLE IF NOT EXISTS tdc_clearance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    drug_id     TEXT,
    drug_name   TEXT,
    smiles      TEXT,                      -- molecular SMILES string
    y           REAL                       -- clearance value (mL/min/kg)
);

-- Half-life prediction (Obach dataset)
CREATE TABLE IF NOT EXISTS tdc_halflife (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    drug_id     TEXT,
    drug_name   TEXT,
    smiles      TEXT,
    y           REAL                       -- half-life (hours)
);

-- Bioavailability (Ma dataset)
CREATE TABLE IF NOT EXISTS tdc_bioavailability (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    drug_id     TEXT,
    drug_name   TEXT,
    smiles      TEXT,
    y           REAL                       -- bioavailability (0-1)
);

-- Plasma protein binding rate (AstraZeneca dataset)
CREATE TABLE IF NOT EXISTS tdc_ppbr (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    drug_id     TEXT,
    drug_name   TEXT,
    smiles      TEXT,
    y           REAL                       -- binding rate (%)
);

-- =============================================================
-- INDEXES for common query patterns
-- =============================================================
CREATE INDEX IF NOT EXISTS idx_groups_study ON pkdb_groups(study_sid);
CREATE INDEX IF NOT EXISTS idx_groups_measurement ON pkdb_groups(measurement_type);
CREATE INDEX IF NOT EXISTS idx_groups_substance ON pkdb_groups(substance);

CREATE INDEX IF NOT EXISTS idx_individuals_study ON pkdb_individuals(study_sid);
CREATE INDEX IF NOT EXISTS idx_individuals_measurement ON pkdb_individuals(measurement_type);
CREATE INDEX IF NOT EXISTS idx_individuals_pk ON pkdb_individuals(individual_pk);

CREATE INDEX IF NOT EXISTS idx_interventions_study ON pkdb_interventions(study_sid);
CREATE INDEX IF NOT EXISTS idx_interventions_substance ON pkdb_interventions(substance);

CREATE INDEX IF NOT EXISTS idx_tdc_clearance_smiles ON tdc_clearance(smiles);
CREATE INDEX IF NOT EXISTS idx_tdc_halflife_smiles ON tdc_halflife(smiles);

-- =============================================================
-- VIEWS for common analysis queries
-- =============================================================

-- Pivot groups to get one row per group with age, weight, sex
CREATE VIEW IF NOT EXISTS v_group_demographics AS
SELECT 
    g.study_sid,
    g.study_name,
    g.group_pk,
    g.group_name,
    g.group_count,
    MAX(CASE WHEN g.measurement_type = 'age' THEN g.mean END) AS age_mean,
    MAX(CASE WHEN g.measurement_type = 'age' THEN g.sd END) AS age_sd,
    MAX(CASE WHEN g.measurement_type = 'age' THEN g.unit END) AS age_unit,
    MAX(CASE WHEN g.measurement_type = 'weight' THEN g.mean END) AS weight_mean,
    MAX(CASE WHEN g.measurement_type = 'weight' THEN g.sd END) AS weight_sd,
    MAX(CASE WHEN g.measurement_type = 'weight' THEN g.unit END) AS weight_unit,
    MAX(CASE WHEN g.measurement_type = 'bmi' THEN g.mean END) AS bmi_mean,
    MAX(CASE WHEN g.measurement_type = 'height' THEN g.mean END) AS height_mean,
    MAX(CASE WHEN g.measurement_type = 'sex' THEN g.choice END) AS sex,
    MAX(CASE WHEN g.measurement_type = 'smoking' THEN g.choice END) AS smoking,
    MAX(CASE WHEN g.measurement_type = 'oral contraceptives' THEN g.choice END) AS oral_contraceptives,
    MAX(CASE WHEN g.measurement_type = 'species' THEN g.choice END) AS species,
    MAX(CASE WHEN g.measurement_type = 'healthy' THEN g.choice END) AS healthy
FROM pkdb_groups g
GROUP BY g.study_sid, g.study_name, g.group_pk, g.group_name, g.group_count;

-- Pivot individuals similarly
CREATE VIEW IF NOT EXISTS v_individual_demographics AS
SELECT
    i.study_sid,
    i.study_name,
    i.individual_pk,
    i.individual_name,
    i.individual_group_pk,
    MAX(CASE WHEN i.measurement_type = 'age' THEN i.value END) AS age,
    MAX(CASE WHEN i.measurement_type = 'weight' THEN i.value END) AS weight_kg,
    MAX(CASE WHEN i.measurement_type = 'height' THEN i.value END) AS height_cm,
    MAX(CASE WHEN i.measurement_type = 'bmi' THEN i.value END) AS bmi,
    MAX(CASE WHEN i.measurement_type = 'sex' THEN i.choice END) AS sex,
    MAX(CASE WHEN i.measurement_type = 'smoking' THEN i.choice END) AS smoking,
    MAX(CASE WHEN i.measurement_type = 'ethnicity' THEN i.choice END) AS ethnicity,
    MAX(CASE WHEN i.measurement_type LIKE '%genotype%' THEN i.choice END) AS genotype,
    MAX(CASE WHEN i.measurement_type LIKE '%phenotype%' THEN i.choice END) AS phenotype
FROM pkdb_individuals i
GROUP BY i.study_sid, i.study_name, i.individual_pk, 
         i.individual_name, i.individual_group_pk;

-- Intervention summary: one row per intervention with dose info
CREATE VIEW IF NOT EXISTS v_intervention_doses AS
SELECT
    study_sid,
    study_name,
    intervention_pk,
    name AS intervention_name,
    substance,
    route,
    form,
    application,
    value AS dose_value,
    mean AS dose_mean,
    unit AS dose_unit,
    time AS dosing_times,
    time_unit
FROM pkdb_interventions
WHERE measurement_type = 'dosing';