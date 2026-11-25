"""端到端: 事件溯源、更正复现、虚构样例诊断与参数回收。"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ordinal_app import diagnostics, report, synth
from ordinal_app.fit import fit_model
from ordinal_app.ingest import parse_csv
from ordinal_app.stats import FitConfig
from ordinal_app.store import EventStore


class TestEventSourcing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EventStore(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _load_base(self, seed=7):
        ds = synth.make_base(seed=seed)
        self.store.import_batch(
            ds.rows, ds.categories, batch_id="b1",
            note=json.dumps(ds.generating, ensure_ascii=False))
        return ds

    def test_import_and_projection_keeps_raw_text(self):
        ds = self._load_base()
        proj = self.store.project()
        self.assertEqual(proj.categories, ds.categories)
        self.assertEqual(len(proj.records), len(ds.rows))
        # 原文本逐字保留
        self.assertIn(proj.records[0].raw_label, ds.categories)
        self.assertFalse(proj.records[0].corrected)

    def test_correction_is_append_only_and_reproducible(self):
        self._load_base()
        rep1 = report.build_report(self.store, kind="fixed",
                                   include_selection=False)
        v_before = rep1["version_id"]
        target = self.store.project().records[3]
        old_label = target.effective_label
        other = [c for c in self.store.project().categories
                 if c != old_label][0]
        # 追加更正
        self.store.correct_rating(target.event_id, other,
                                  reason="录入误点")
        # 事件日志只增不减
        n_events_after = len(self.store.events)
        self.assertGreater(n_events_after, rep1["n_events"])
        proj = self.store.project()
        rec = next(r for r in proj.records
                   if r.event_id == target.event_id)
        self.assertEqual(rec.raw_label, old_label)        # 原文本保留
        self.assertEqual(rec.effective_label, other)      # 生效文本更新
        self.assertTrue(rec.corrected)

        # 旧版本仍可投影, 旧报告可复现
        old_proj = self.store.project(n_events=rep1["n_events"],
                                      version_id=v_before)
        old_rec = next(r for r in old_proj.records
                       if r.event_id == target.event_id)
        self.assertEqual(old_rec.effective_label, old_label)
        check = report.reproduce_report(
            self.store, rep1["report_id"])
        self.assertTrue(check["reproduced"])
        self.assertFalse(check["valid_now"])

        # 报告文件本身不可变且仍在磁盘上
        saved = self.store.load_report(rep1["report_id"])
        self.assertEqual(saved["version_id"], v_before)

    def test_duplicate_batch_rejected(self):
        self._load_base()
        with self.assertRaises(Exception):
            self.store.import_batch(
                [{"project": "X", "rater": "Y", "period": 0,
                  "raw_label": "一般"}],
                self.store.categories(), batch_id="b1")

    def test_version_chain_is_linear(self):
        self._load_base()
        self.store.correct_rating(
            self.store.project().records[0].event_id,
            self.store.project().categories[-1], reason="x")
        versions = self.store.versions()
        self.assertGreaterEqual(len(versions), 2)
        ids = [v.version_id for v in versions]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIsNone(versions[0].parent_id)
        for a, b in zip(versions, versions[1:]):
            self.assertEqual(a.version_id, b.parent_id)


class TestSyntheticScenarios(unittest.TestCase):
    def _fit_store(self, ds, batch="b"):
        tmp = tempfile.TemporaryDirectory()
        store = EventStore(tmp.name)
        store.import_batch(ds.rows, ds.categories, batch_id=batch)
        return store, tmp

    def test_base_parameter_recovery(self):
        ds = synth.make_base(seed=7)
        store, tmp = self._fit_store(ds)
        self.addCleanup(tmp.cleanup)
        fit = fit_model(store.project(), FitConfig(kind="drift"))
        self.assertTrue(fit.converged)
        gen_b = ds.generating["abilities"]
        # 零和锚下比较: 去均值后的相关与尺度
        gen = np.array([gen_b[p] for p in fit.projects])
        est = fit.abilities
        corr = np.corrcoef(gen - gen.mean(), est - est.mean())[0, 1]
        # 有序类别观测噪声大; 排序恢复 (相关 >=0.85) 是小样本下的现实目标
        self.assertGreater(corr, 0.85, f"能力回收相关过低: {corr}")
        # 能力尺度不应被先验压缩得失真
        self.assertLess(est.std() / max(gen.std(), 1e-9), 1.5)
        # 阈值严格有序
        self.assertTrue((np.diff(fit.thresholds, axis=-1) > 0).all())

    def test_separation_detected(self):
        ds = synth.make_separation()
        store, tmp = self._fit_store(ds)
        self.addCleanup(tmp.cleanup)
        proj = store.project()
        comps = diagnostics.overlap_components(proj)
        self.assertEqual(len(comps), 2)
        diag = diagnostics.diagnose(proj)
        self.assertTrue(diag.separated)
        kinds = {p["kind"] for p in diag.problems}
        self.assertIn("separation", kinds)
        # 拟合仍可运行 (先验兜底), 但报告带警告
        fit = fit_model(proj, FitConfig(kind="fixed"))
        self.assertEqual(fit.thresholds.shape[0], 4)

    def test_missing_category_detected(self):
        ds = synth.make_missing_category()
        store, tmp = self._fit_store(ds)
        self.addCleanup(tmp.cleanup)
        proj = store.project()
        usage = diagnostics.category_usage(proj)
        self.assertIn("R2_sparse",
                      {r for r, m in usage["missing_by_rater"].items() if m})
        diag = diagnostics.diagnose(proj)
        kinds = {p["kind"] for p in diag.problems}
        self.assertTrue(
            {"missing_category", "missing_category_rater"} & kinds)

    def test_reversal_detected(self):
        ds = synth.make_reversal()
        store, tmp = self._fit_store(ds)
        self.addCleanup(tmp.cleanup)
        orient = diagnostics.rater_orientation(store.project())
        rev = [o for o in orient if o["rater"] == "R3_rev"][0]
        self.assertTrue(rev["reversed"])

    def test_probabilities_sum_to_one_and_use_text_labels(self):
        ds = synth.make_base(seed=3)
        store, tmp = self._fit_store(ds)
        self.addCleanup(tmp.cleanup)
        fit = fit_model(store.project(), FitConfig(kind="fixed"))
        table = fit.probability_table("logit")
        for row in table[:10]:
            s = sum(row[f"P({c})"] for c in ds.categories)
            self.assertAlmostEqual(s, 1.0, places=9)
            for c in ds.categories:
                self.assertIn(f"P({c})", row)


class TestCsvParsing(unittest.TestCase):
    def test_parse_with_header_directive(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            p.write_text(
                "# categories,很差,一般,满意\n"
                "project,rater,period,raw_label\n"
                "P0,R0,1,满意\nP1,R0,1,很差\n", encoding="utf-8")
            rows, cats = parse_csv(p)
            self.assertEqual(cats, ["很差", "一般", "满意"])
            self.assertEqual(rows[0]["raw_label"], "满意")
            self.assertEqual(rows[1]["period"], 1)

    def test_reject_unknown_label(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            p.write_text(
                "# categories,很差,满意\n"
                "project,rater,period,raw_label\n"
                "P0,R0,1,一般\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_csv(p)


if __name__ == "__main__":
    unittest.main()
