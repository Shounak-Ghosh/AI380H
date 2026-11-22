-- Catalog and observed units for candidate measurements, including other fluids
-- and methods. The extraction uses the verified allowlist in features.py.
WITH candidates AS (
  SELECT itemid, label, fluid, category
  FROM `physionet-data.mimiciv_3_1_hosp.d_labitems`
  WHERE REGEXP_CONTAINS(
    LOWER(label),
    r'^(white blood cells|hemoglobin|platelet count|sodium|potassium|creatinine|urea nitrogen|glucose|bicarbonate)(\b|$)'
  )
), profile AS (
  SELECT l.itemid, l.valueuom, COUNT(*) AS measurements,
    COUNTIF(l.valuenum IS NOT NULL AND NOT IS_NAN(l.valuenum)
      AND NOT IS_INF(l.valuenum)) AS numeric_measurements,
    COUNTIF(l.hadm_id IS NULL) AS missing_hadm_id,
    COUNTIF(l.charttime IS NULL) AS missing_charttime,
    COUNTIF(l.storetime IS NULL) AS missing_storetime
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` AS l
  INNER JOIN candidates USING (itemid)
  GROUP BY l.itemid, l.valueuom
)
SELECT c.itemid, c.label, c.fluid, c.category, p.valueuom,
  COALESCE(p.measurements, 0) AS measurements,
  COALESCE(p.numeric_measurements, 0) AS numeric_measurements,
  COALESCE(p.missing_hadm_id, 0) AS missing_hadm_id,
  COALESCE(p.missing_charttime, 0) AS missing_charttime,
  COALESCE(p.missing_storetime, 0) AS missing_storetime
FROM candidates AS c
LEFT JOIN profile AS p USING (itemid)
ORDER BY c.label, c.fluid, c.category, c.itemid, measurements DESC, p.valueuom
