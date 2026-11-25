"""小数据似然与解析梯度核对 (无第三方测试依赖, unittest)。

tiny 夹具: 3 项目能力 b=[-1,0,1], 2 评价者阈值
R0: [-0.5, 1.0], R1: [0.2, 1.6], 3 等级 [很差, 一般, 满意], logit 链接。
所有概率均可手工复算。
"""
import unittest

import numpy as np
from scipy.special import expit

from ordinal_app import stats
from ordinal_app.stats import (FitConfig, ParamLayout, TensorData,
                               log_likelihood_natural,
                               neg_log_posterior,
                               neg_log_posterior_grad, numerical_gradient)
from ordinal_app.store import Projection
from ordinal_app.models import RatingRecord


CATS = ["很差", "一般", "满意"]
ABILITIES = np.array([-1.0, 0.0, 1.0])
THRESHOLDS = np.array([[[-0.5, 1.0]], [[0.2, 1.6]]])


def _probs_hand(ability, tau):
    """3 等级 logit 概率手工公式。"""
    p0 = expit(tau[0] - ability)
    p2 = 1.0 - expit(tau[1] - ability)
    p1 = 1.0 - p0 - p2
    return np.array([p0, p1, p2])


class TestTinyLikelihood(unittest.TestCase):
    def setUp(self):
        # 每个 (项目, 评价者) 一条观测, 等级取模态
        records = []
        eid = 0
        for ri, r in enumerate(["R0", "R1"]):
            for pi, p in enumerate(["P0", "P1", "P2"]):
                probs = _probs_hand(ABILITIES[pi], THRESHOLDS[ri, 0])
                y = int(np.argmax(probs))
                records.append(RatingRecord(
                    event_id=f"e{eid}", project=p, rater=r, period=0,
                    raw_label=CATS[y], effective_label=CATS[y],
                    corrected=False))
                eid += 1
        self.proj = Projection(CATS, records, "v-test", n_events=6,
                               n_corrections=0)
        self.data = TensorData.from_projection(self.proj)

    def test_hand_computed_probs(self):
        # P(Y=很差 | P0,R0): expit(-0.5 - (-1)) = expit(0.5)
        self.assertAlmostEqual(_probs_hand(-1.0, [-0.5, 1.0])[0],
                               expit(0.5), places=12)
        # P(Y=满意 | P2,R1): 1 - expit(1.6-1)=1-expit(0.6)
        self.assertAlmostEqual(_probs_hand(1.0, [0.2, 1.6])[2],
                               1 - expit(0.6), places=12)

    def test_model_probs_match_hand(self):
        probs = stats.category_probs(
            ABILITIES, THRESHOLDS,
            self.data.project_ix, self.data.rater_ix,
            self.data.period_ix, "logit")
        for n, rec in enumerate(self.proj.records):
            pi = ["P0", "P1", "P2"].index(rec.project)
            ri = ["R0", "R1"].index(rec.rater)
            hand = _probs_hand(ABILITIES[pi], THRESHOLDS[ri, 0])
            np.testing.assert_allclose(probs[n], hand, atol=1e-12)
            self.assertAlmostEqual(probs[n].sum(), 1.0, places=12)

    def test_loglikelihood_explicit_sum(self):
        probs = stats.category_probs(
            ABILITIES, THRESHOLDS,
            self.data.project_ix, self.data.rater_ix,
            self.data.period_ix, "logit")
        hand_ll = sum(np.log(probs[i, self.data.y[i]])
                      for i in range(len(self.data.y)))
        self.assertAlmostEqual(
            log_likelihood_natural(ABILITIES, THRESHOLDS, self.data, "logit"),
            hand_ll, places=10)

    def test_probit_matches_known_cdf(self):
        from scipy.stats import norm
        probs = stats.category_probs(
            ABILITIES, THRESHOLDS,
            self.data.project_ix, self.data.rater_ix,
            self.data.period_ix, "probit")
        p00 = norm.cdf(-0.5 - (-1.0))
        self.assertAlmostEqual(probs[0, 0], p00, places=12)


class TestGradient(unittest.TestCase):
    def _make_data(self, seed=3, drift=False):
        rng = np.random.default_rng(seed)
        records = []
        eid = 0
        periods = [0, 1, 2] if drift else [0]
        for _ in range(60):
            p = f"P{rng.integers(0, 4)}"
            r = f"R{rng.integers(0, 3)}"
            t = int(rng.choice(periods))
            y = int(rng.integers(0, 3))
            records.append(RatingRecord(
                event_id=f"e{eid}", project=p, rater=r, period=t,
                raw_label=CATS[y], effective_label=CATS[y], corrected=False))
            eid += 1
        proj = Projection(CATS, records, "v", n_events=60, n_corrections=0)
        return proj, TensorData.from_projection(proj)

    def test_analytic_vs_numeric_fixed(self):
        proj, data = self._make_data()
        layout = ParamLayout(4, 3, 1, 3, FitConfig(kind="fixed"))
        theta = layout.initial_theta()
        g = neg_log_posterior_grad(theta, layout, data, "logit")
        gn = numerical_gradient(theta, layout, data, "logit")
        np.testing.assert_allclose(g, gn, atol=1e-5, rtol=1e-5)

    def test_analytic_vs_numeric_drift(self):
        proj, data = self._make_data(drift=True)
        layout = ParamLayout(4, 3, 3, 3,
                             FitConfig(kind="drift", sigma_walk=0.3))
        theta = layout.initial_theta()
        g = neg_log_posterior_grad(theta, layout, data, "logit")
        gn = numerical_gradient(theta, layout, data, "logit")
        np.testing.assert_allclose(g, gn, atol=1e-5, rtol=1e-5)

    def test_thresholds_never_cross(self):
        proj, data = self._make_data(drift=True)
        layout = ParamLayout(4, 3, 3, 3, FitConfig(kind="drift"))
        rng = np.random.default_rng(0)
        for _ in range(20):
            theta = rng.normal(0, 3, layout.n_params)
            _, tau = layout.unpack(theta)
            diffs = tau[..., 1:] - tau[..., :1]
            self.assertTrue((diffs > 0).all(),
                            "阈值必须严格有序, 发现交叉")


if __name__ == "__main__":
    unittest.main()
