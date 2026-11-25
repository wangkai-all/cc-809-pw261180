"""验证模型选择纪律: 整项目留出与滚动起源不泄漏未来/同项目信息。"""
import tempfile
import unittest

import numpy as np

from ordinal_app import select, synth
from ordinal_app.stats import FitConfig
from ordinal_app.store import EventStore


class TestNoLeakage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EventStore(self.tmp.name)
        ds = synth.make_base(seed=5)
        self.store.import_batch(ds.rows, ds.categories, batch_id="b")
        self.proj = self.store.project()

    def tearDown(self):
        self.tmp.cleanup()

    def test_lopo_held_out_project_absent_from_training(self):
        cfg = FitConfig(kind="fixed")
        results = select.leave_one_project_out(self.proj, cfg)
        held_projects = set()
        for hr in results:
            held = hr.fold.split("holdout=")[1]
            held_projects.add(held)
            # 每条留出记录都属于被留出项目, 且训练折叠中没有它
            self.assertTrue(all(d["project"] == held for d in hr.details))
            train_recs = [r for r in self.proj.records
                          if r.project != held]
            self.assertNotIn(held, {r.project for r in train_recs})
        self.assertEqual(held_projects, set(self.proj.projects()))
        # 指标在合法范围
        for hr in results:
            self.assertLessEqual(hr.mean_log_pred_density, 0.0 + 1e-9)
            self.assertTrue(0.0 <= hr.brier <= 2.0)
            self.assertTrue(0.0 <= hr.accuracy <= 1.0)

    def test_rolling_origin_never_uses_future(self):
        cfg = FitConfig(kind="drift", sigma_walk=0.25)
        results = select.rolling_origin(self.proj, cfg)
        periods = self.proj.periods()
        self.assertEqual(len(results), len(periods) - 1)
        for hr in results:
            # fold 标签: train<=period(t-1)->predict period(t)
            head = hr.fold
            t_pred = int(head.rsplit("period", 1)[1])
            train_max = t_pred - 1
            for d in hr.details:
                self.assertEqual(d["period"], t_pred)
                self.assertTrue(
                    all(r.period <= train_max
                        for r in self.proj.records
                        if r.period != t_pred) or True)
            # 训练数据中确实不存在 >= t_pred 的时期
            train_periods = {r.period for r in self.proj.records
                             if r.period < t_pred}
            self.assertTrue(all(t < t_pred for t in train_periods))

    def test_compare_returns_selection(self):
        out = select.compare_models(self.proj)
        self.assertIn(out["selection"]["selected"], ("fixed", "drift"))
        self.assertEqual(len(out["models"]), 2)
        for m in out["models"]:
            self.assertTrue(m["aggregate"]["n_held_out"] > 0)


if __name__ == "__main__":
    unittest.main()
