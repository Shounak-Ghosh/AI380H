import math
from pathlib import Path
import sqlite3
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cohort import LABEL, admission_age, build_cohort, partition, validate_cohort
from features import LAB_ITEMS, assemble


START = pd.Timestamp("2150-01-02 10:00")
DISCHARGE = START + pd.Timedelta("1d")


def admission(hadm_id, start=START, end=None, kind="EW EMER.", **overrides):
    return {
        "subject_id": 7, "hadm_id": hadm_id, "admittime": start,
        "dischtime": end if end is not None else start + pd.Timedelta("1d"),
        "deathtime": None, "hospital_expire_flag": 0,
        "admission_type": kind, "admission_location": "EMERGENCY ROOM",
        "discharge_location": "HOME", "insurance": "Other", **overrides,
    }


def patients(dod=None):
    return pd.DataFrame([{"subject_id": 7, "anchor_age": 17, "anchor_year": 2149,
                          "gender": "F", "dod": dod}])


class CohortTests(unittest.TestCase):
    def build(self, *rows, dod=None):
        result, flow = build_cohort(pd.DataFrame(rows), patients(dod))
        validate_cohort(result)
        self.assertEqual(flow.admissions.sum(), len(rows))
        return result.set_index("hadm_id"), flow.set_index("disposition").admissions

    def test_age_and_los_boundaries(self):
        ages = admission_age(pd.Series([17, 17]), pd.Series([2150, 2149]),
                             pd.Series([START, START]))
        self.assertEqual(ages.tolist(), [17, 18])
        data, _ = self.build(admission(1))
        self.assertEqual(data.loc[1, "age"], 18)
        self.assertEqual(data.loc[1, "prediction_time"], DISCHARGE)
        younger, flow = build_cohort(pd.DataFrame([admission(1)]),
                                    patients().assign(anchor_year=2150))
        self.assertTrue(younger.empty)
        self.assertEqual(flow.set_index("disposition").loc["under_18", "admissions"], 1)
        _, flow = self.build(admission(1, end=DISCHARGE - pd.Timedelta("1s")))
        self.assertEqual(flow["stay_under_24h"], 1)
        _, flow = self.build(admission(1, hospital_expire_flag=1))
        self.assertEqual(flow["in_hospital_death_or_unknown"], 1)

    def test_order_and_30_day_boundary(self):
        for lag, label in [(pd.Timedelta("30d"), 1),
                           (pd.Timedelta("30d 1s"), 0)]:
            with self.subTest(lag=lag):
                data, _ = self.build(admission(2, DISCHARGE + lag), admission(1))
                self.assertEqual(data.loc[1, LABEL], label)

    def test_same_day_and_same_instant(self):
        data, _ = self.build(admission(1), admission(2, DISCHARGE + pd.Timedelta("1min")))
        self.assertEqual(data.loc[1, LABEL], 1)
        data, flow = self.build(admission(1), admission(2, DISCHARGE))
        self.assertNotIn(1, data.index)
        self.assertEqual(flow["same_instant_return"], 1)

    def test_planned_and_transfer_compete_with_later_unplanned(self):
        for followup, exclusion in [
            ({"kind": "ELECTIVE"}, "competing_planned"),
            ({"admission_location": "TRANSFER FROM HOSPITAL"}, "competing_transfer"),
            ({"kind": "DIRECT OBSERVATION"}, "ambiguous_return"),
        ]:
            with self.subTest(exclusion=exclusion):
                data, flow = self.build(admission(1),
                                        admission(2, DISCHARGE + pd.Timedelta("1d"), **followup),
                                        admission(3, DISCHARGE + pd.Timedelta("5d")))
                self.assertNotIn(1, data.index)
                self.assertEqual(flow[exclusion], 1)

    def test_nested_overlaps_and_invalid_timelines(self):
        data, flow = self.build(admission(1, end=START + pd.Timedelta("8d")),
                                admission(2, START + pd.Timedelta("1d")),
                                admission(3, START + pd.Timedelta("3d")))
        self.assertTrue(data.empty)
        self.assertEqual(flow["overlapping_index_admission"], 3)
        _, flow = self.build(admission(1), admission(2, pd.NaT))
        self.assertEqual(flow["invalid_admission_interval"], 1)
        self.assertEqual(flow["incomplete_patient_timeline"], 1)
        _, flow = self.build(admission(1), admission(1), admission(3, START + pd.Timedelta("5d")))
        self.assertEqual(flow["duplicate_admission_id"], 2)
        self.assertEqual(flow["incomplete_patient_timeline"], 1)

    def test_history_and_known_death(self):
        data, _ = self.build(admission(3, START + pd.Timedelta("20d")),
                             admission(2, START + pd.Timedelta("3d"), kind="ELECTIVE"),
                             admission(1, end=START + pd.Timedelta("2h")))
        self.assertEqual(data.loc[2, "prior_admissions"], 1)
        self.assertEqual(data.loc[3, "prior_admissions"], 2)
        self.assertEqual(data.loc[3, "days_since_previous_discharge"], 16)
        data, _ = self.build(admission(1))
        self.assertEqual(data.loc[1, "prior_admissions"], 0)
        self.assertTrue(pd.isna(data.loc[1, "days_since_previous_discharge"]))
        _, flow = self.build(admission(1), dod="2150-01-15")
        self.assertEqual(flow["death_before_30d"], 1)

    def test_partitions_are_patient_disjoint_and_order_invariant(self):
        data = pd.DataFrame({"subject_id": [i // 2 for i in range(200)], "hadm_id": range(200)})
        splits = partition(data)
        self.assertTrue(splits.groupby("subject_id").partition.nunique().eq(1).all())
        self.assertEqual(splits.partition.value_counts().to_dict(),
                         {"train": 140, "validation": 30, "test": 30})
        pd.testing.assert_frame_equal(splits, partition(data.iloc[::-1]))


class LabSQLTests(unittest.TestCase):
    def test_first_valid_value_and_missingness(self):
        # Run the actual ranking, filters and pivot offline. Adapt only the
        # table references, parameter table, and timestamp addition syntax.
        with sqlite3.connect(":memory:") as db:
            db.create_function("IS_NAN", 1, lambda x: x is not None and math.isnan(x))
            db.create_function("IS_INF", 1, lambda x: x is not None and math.isinf(x))
            db.executescript("""
                CREATE TABLE admissions (subject_id INTEGER, hadm_id INTEGER, admittime TEXT,
                    dischtime TEXT, hospital_expire_flag INTEGER, deathtime TEXT);
                CREATE TABLE labevents (subject_id INTEGER, hadm_id INTEGER, itemid INTEGER,
                    labevent_id INTEGER, charttime TEXT, storetime TEXT, valueuom TEXT, valuenum REAL);
                CREATE TABLE lab_items (feature TEXT, itemid INTEGER, unit TEXT, allow_zero INTEGER);
                INSERT INTO admissions VALUES
                    (7, 1, '2150-01-02 10:00:00', '2150-01-03 10:00:00', 0, NULL),
                    (7, 2, '2150-01-05 10:00:00', '2150-01-06 10:00:00', 0, NULL);
            """)
            db.executemany("INSERT INTO lab_items VALUES (?,?,?,?)",
                           [(name, item, unit, zero) for name, item, _, _, unit, zero in LAB_ITEMS])

            def event(id_, chart, store, value, unit="K/uL", subject=7, hadm=1, item=51301):
                return subject, hadm, item, id_, chart, store, unit, value

            valid = "2150-01-02 12:00:00"
            cutoff = "2150-01-03 10:00:00"
            pre = "2150-01-02 10:00:00"
            later = "2150-01-03 10:00:01"
            rows = [
                event(2, valid, valid, 9), event(1, valid, valid, 8, " k/ul "),
                event(3, pre, later, 99), event(4, pre, None, 99),
                event(5, pre, pre, None), event(6, pre, pre, -1),
                event(7, pre, pre, 5, "other"), event(8, pre, pre, 22, subject=9),
                event(9, pre, pre, 22, hadm=2), event(10, later, later, 22),
                event(11, cutoff, cutoff, 140, "mEq/L", item=50983),
                event(12, valid, pre, 22), event(13, pre, pre, float("inf")),
                event(14, "2150-01-02 09:59:59", pre, 22),
            ]
            db.executemany("INSERT INTO labevents VALUES (?,?,?,?,?,?,?,?)", rows)
            sql_path = Path(__file__).resolve().parents[1] / "src/sql/first_24h_labs.sql"
            sql = (sql_path.read_text().replace("physionet-data.mimiciv_3_1_hosp.", "")
                   .replace("UNNEST(@lab_items)", "lab_items").replace(
                       "DATETIME_ADD(admittime, INTERVAL 24 HOUR)", "datetime(admittime, '+24 hours')"
                   ))
            labs = pd.read_sql_query(sql, db)
        index, _ = build_cohort(pd.DataFrame([admission(1),
                                              admission(2, START + pd.Timedelta("3d"))]), patients())
        data = assemble(index, labs).set_index("hadm_id")
        self.assertEqual(data.loc[1, "wbc"], 8)
        self.assertEqual(data.loc[1, "sodium"], 140)
        self.assertEqual(data.loc[1, "wbc_missing"], 0)
        self.assertTrue(pd.isna(data.loc[2, "wbc"]))
        self.assertEqual(data.loc[2, "wbc_missing"], 1)
        self.assertTrue(pd.isna(data.loc[1, "bicarbonate"]))


if __name__ == "__main__":
    unittest.main()
