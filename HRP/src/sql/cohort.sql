-- One source row per admission record. Keep all encounters for the timeline;
-- adult/LOS/death restrictions apply at index admission selection in cohort.py.
SELECT subject_id, hadm_id, admittime, dischtime, deathtime,
  admission_type, admission_location, discharge_location, insurance, hospital_expire_flag
FROM `physionet-data.mimiciv_3_1_hosp.admissions`
