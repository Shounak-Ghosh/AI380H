-- @lab_items is ARRAY<STRUCT<feature STRING, itemid INT64, unit STRING,
--                           allow_zero BOOL>>.
-- Return measured admissions. The final cohort left join retains all others.
WITH admission_windows AS (
  SELECT subject_id, hadm_id, admittime,
    DATETIME_ADD(admittime, INTERVAL 24 HOUR) AS prediction_time
  FROM `physionet-data.mimiciv_3_1_hosp.admissions`
  WHERE dischtime >= DATETIME_ADD(admittime, INTERVAL 24 HOUR)
    AND hospital_expire_flag = 0 AND deathtime IS NULL
), eligible_observations AS (
  SELECT a.hadm_id, m.feature, l.labevent_id, l.valuenum, l.charttime, l.storetime
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` AS l
  INNER JOIN UNNEST(@lab_items) AS m ON l.itemid = m.itemid
  INNER JOIN admission_windows AS a
    ON l.hadm_id = a.hadm_id AND l.subject_id = a.subject_id
  WHERE l.charttime BETWEEN a.admittime AND a.prediction_time
    AND l.storetime BETWEEN l.charttime AND a.prediction_time
    AND LOWER(TRIM(l.valueuom)) = LOWER(m.unit)
    AND (l.valuenum > 0 OR (m.allow_zero AND l.valuenum = 0))
    AND NOT IS_NAN(l.valuenum) AND NOT IS_INF(l.valuenum)
    AND l.labevent_id IS NOT NULL
), ranked AS (
  SELECT hadm_id, feature, valuenum, charttime, storetime,
    ROW_NUMBER() OVER (
      PARTITION BY hadm_id, feature
      ORDER BY charttime, storetime, labevent_id, valuenum
    ) AS observation_order
  FROM eligible_observations
)
SELECT hadm_id,
  MAX(CASE WHEN feature = 'wbc' THEN valuenum END) AS wbc,
  MAX(CASE WHEN feature = 'hemoglobin' THEN valuenum END) AS hemoglobin,
  MAX(CASE WHEN feature = 'platelets' THEN valuenum END) AS platelets,
  MAX(CASE WHEN feature = 'sodium' THEN valuenum END) AS sodium,
  MAX(CASE WHEN feature = 'potassium' THEN valuenum END) AS potassium,
  MAX(CASE WHEN feature = 'creatinine' THEN valuenum END) AS creatinine,
  MAX(CASE WHEN feature = 'bun' THEN valuenum END) AS bun,
  MAX(CASE WHEN feature = 'glucose' THEN valuenum END) AS glucose,
  MAX(CASE WHEN feature = 'bicarbonate' THEN valuenum END) AS bicarbonate,
  MIN(charttime) AS first_lab_charttime,
  MAX(charttime) AS last_lab_charttime,
  MAX(storetime) AS last_lab_storetime,
  COUNT(*) AS selected_labs
FROM ranked
WHERE observation_order = 1
GROUP BY hadm_id
