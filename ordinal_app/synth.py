"""虚构数据生成: 所有样例都带 *已知生成参数*, 供参数回收与小数据似然核对。

生成模型与 stats.py 完全一致::

    P(Y=k | p,r,t) = F(tau_{r,t,k} - b_p) - F(tau_{r,t,k-1} - b_p)

提供的样例:
  base       正常的项目/评价者/时期结构, 阈值随时期轻微漂移
  reversal   一名评价者尺度反转 (阈值为正常模板镜像, 等级映射近似 k->K-1-k)
  separation 两组项目完全不共享评价者 -> 重叠图不连通
  missing    一名评价者从不使用中间等级 -> 缺等级边界情形
  tiny       极小数据集, 手工可核对的似然测试夹具
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .stats import link_funs

DEFAULT_CATEGORIES = ["很差", "较差", "一般", "满意", "很满意"]


@dataclass
class SyntheticDataset:
    rows: list[dict[str, Any]]
    categories: list[str]
    generating: dict[str, Any]
    scenario: str

    def to_dict(self) -> dict[str, Any]:
        return {"scenario": self.scenario,
                "categories": self.categories,
                "rows": self.rows,
                "generating": self.generating}


def _sample_y(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(probs), p=probs / probs.sum()))


def _make_rows(abilities, thresholds, projects, raters, periods, link,
               design: list[tuple[str, str, int]], rng, categories,
               flip_rater: str | None = None
               ) -> list[dict[str, Any]]:
    F, _, _, _ = link_funs(link)
    rows = []
    p_ix = {p: i for i, p in enumerate(projects)}
    r_ix = {r: i for i, r in enumerate(raters)}
    t_ix = {t: i for i, t in enumerate(periods)}
    K = len(categories)
    for p, r, t in design:
        b = abilities[p_ix[p]]
        tau = thresholds[r_ix[r], t_ix[t]]
        z = np.clip(tau - b, -30, 30)
        cum = F(z)
        probs = np.empty(K)
        probs[0] = cum[0]
        probs[1:K - 1] = cum[1:] - cum[:-1]
        probs[K - 1] = 1 - cum[-1]
        probs = np.clip(probs, 1e-12, 1)
        probs /= probs.sum()
        k = _sample_y(probs, rng)
        latent_k = k
        if flip_rater is not None and r == flip_rater:
            # 尺度反转: 该评价者把好/坏方向颠倒, 记录等级 k -> K-1-k
            k = K - 1 - k
        rows.append({"project": p, "rater": r, "period": t,
                     "raw_label": categories[k],
                     "true_category_index": k,
                     "latent_category_index": latent_k,
                     "flipped": bool(flip_rater is not None
                                     and r == flip_rater),
                     "true_probability": float(probs[latent_k])})
    return rows


def _fixed_thresholds(raters, K, rng, spread=1.4):
    """生成严格有序的固定阈值。"""
    R = len(raters)
    tau = np.zeros((R, 1, K - 1))
    for r in range(R):
        center = rng.normal(0, 0.7)
        gaps = rng.uniform(0.6, spread, size=K - 2) if K > 2 else np.array([])
        loc = np.concatenate([[0.0], np.cumsum(gaps)])
        tau[r, 0, :] = center + loc - loc.mean()
    return tau


def _shared_template_thresholds(raters, K, rng, jitter=0.15):
    """正常评价者共享同一阈值模板, 只做整体偏移 (保证排序判断可比)。"""
    base_gaps = rng.uniform(0.7, 1.1, size=K - 2) if K > 2 else np.array([])
    loc = np.concatenate([[0.0], np.cumsum(base_gaps)])
    template = loc - loc.mean()
    R = len(raters)
    tau = np.zeros((R, 1, K - 1))
    for r in range(R):
        tau[r, 0, :] = template + rng.normal(0, jitter)
    return tau, template


def make_base(seed: int = 7, link: str = "logit") -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    categories = list(DEFAULT_CATEGORIES)
    K = len(categories)
    projects = [f"P{i}" for i in range(8)]
    raters = [f"R{i}" for i in range(4)]
    periods = [0, 1, 2]
    abilities = rng.normal(0, 1.1, size=len(projects))
    tau0 = _fixed_thresholds(raters, K, rng)
    # 缓变: 每评价者每期位置小幅游走
    thresholds = np.repeat(tau0, len(periods), axis=1)
    walk = rng.normal(0, 0.15, size=(len(raters), len(periods)))
    walk[:, 0] = 0.0
    drift = np.cumsum(walk, axis=1)
    thresholds = thresholds + drift[:, :, None]
    # 高密度设计, 保证重叠图连通且参数可回收
    design = []
    for t in periods:
        for r in raters:
            for p in projects:
                if rng.random() < 0.9:
                    design.append((p, r, t))
    rows = _make_rows(abilities, thresholds, projects, raters, periods,
                      link, design, rng, categories)
    return SyntheticDataset(
        rows, categories,
        {"link": link, "seed": seed,
         "abilities": dict(zip(projects, abilities.tolist())),
         "thresholds": {r: thresholds[i].tolist()
                        for i, r in enumerate(raters)},
         "periods": periods},
        scenario="base")


def make_reversal(seed: int = 11, link: str = "logit") -> SyntheticDataset:
    """R3_rev 评价者尺度反转。

    所有评价者共享同一阈值模板 (仅整体偏移), 但 R3_rev 在把潜在等级
    落到纸面时把方向颠倒: 记录等级 = K-1-潜在等级。于是该评价者对
    项目好坏的排序与所有正常评价者严格相反, 可被方向诊断稳定识别,
    也可在模型中表现为一组反向的拟合阈值。
    """
    rng = np.random.default_rng(seed)
    categories = list(DEFAULT_CATEGORIES)
    K = len(categories)
    projects = [f"P{i}" for i in range(10)]
    raters = ["R0", "R1", "R2", "R3_rev"]
    periods = [0, 1]
    # 能力拉宽到约 [-2,2], 减少等级并列, 让方向信号清晰
    abilities = np.linspace(-2.0, 2.0, len(projects))
    tau0, template = _shared_template_thresholds(raters, K, rng)
    thresholds = np.repeat(tau0, len(periods), axis=1)
    design = []
    for t in periods:
        for r in raters:
            for p in projects:
                design.append((p, r, t))
    rows = _make_rows(abilities, thresholds, projects, raters, periods,
                      link, design, rng, categories,
                      flip_rater="R3_rev")
    return SyntheticDataset(
        rows, categories,
        {"link": link, "seed": seed,
         "abilities": dict(zip(projects, abilities.tolist())),
         "thresholds": {r: thresholds[i].tolist()
                        for i, r in enumerate(raters)},
         "reversed_rater": "R3_rev",
         "normal_template": template.tolist(),
         "flip_mapping": {str(k): K - 1 - k for k in range(K)},
         "note": "R3_rev 把潜在等级 k 记为 K-1-k, 项目排序与正常评价者相反",
         "periods": periods},
        scenario="reversal")


def make_separation(seed: int = 23, link: str = "logit") -> SyntheticDataset:
    """两组项目 {P0..P2} 与 {P3..P5} 完全不共享评价者。"""
    rng = np.random.default_rng(seed)
    categories = list(DEFAULT_CATEGORIES)
    K = len(categories)
    projects = [f"P{i}" for i in range(6)]
    raters = ["RA0", "RA1", "RB0", "RB1"]
    periods = [0, 1]
    abilities = rng.normal(0, 1.0, size=len(projects))
    tau0 = _fixed_thresholds(raters, K, rng)
    thresholds = np.repeat(tau0, len(periods), axis=1)
    design = []
    group_a, group_b = projects[:3], projects[3:]
    rater_a, rater_b = raters[:2], raters[2:]
    for t in periods:
        for r in rater_a:
            for p in group_a:
                design.append((p, r, t))
        for r in rater_b:
            for p in group_b:
                design.append((p, r, t))
    rows = _make_rows(abilities, thresholds, projects, raters, periods,
                      link, design, rng, categories)
    return SyntheticDataset(
        rows, categories,
        {"link": link, "seed": seed,
         "abilities": dict(zip(projects, abilities.tolist())),
         "thresholds": {r: thresholds[i].tolist()
                        for i, r in enumerate(raters)},
         "components": [group_a, group_b],
         "note": "两组项目无共同评价者, 尺度完全分离",
         "periods": periods},
        scenario="separation")


def make_missing_category(seed: int = 31,
                          link: str = "logit") -> SyntheticDataset:
    """R2_sparse 的阈值把中间等级 "一般" 的概率压到接近 0。

    阈值设为 [-1.6, -1.4, 3.6, 3.8]: 中间两个阈值间距 5.0, 而项目能力
    集中在 [-1,1], 等级 2 ("一般") 的累积概率差 F(3.6-b)-F(-1.4-b)
    在所有项目上都极小, 生成数据中该等级确定性缺失。
    """
    rng = np.random.default_rng(seed)
    categories = list(DEFAULT_CATEGORIES)
    K = len(categories)
    projects = [f"P{i}" for i in range(6)]
    raters = ["R0", "R1", "R2_sparse"]
    periods = [0, 1]
    abilities = np.linspace(-1.0, 1.0, len(projects))
    tau0 = _fixed_thresholds(raters, K, rng)
    tau0 = _fixed_thresholds(raters, K, rng)
    # P(Y=k)=F(tau_k-b)-F(tau_{k-1}-b); 让中间等级 2 "一般" 缺失,
    # 需让它的两个边界阈值 tau_1, tau_2 几乎重合 (间隙 0.02), 使该
    # 等级概率 ~0.005; 外侧阈值保持正常间距
    tau0[2, 0, :] = np.array([-2.0, -0.01, 0.01, 2.0])
    thresholds = np.repeat(tau0, len(periods), axis=1)
    design = []
    for t in periods:
        for r in raters:
            for p in projects:
                design.append((p, r, t))
    rows = _make_rows(abilities, thresholds, projects, raters, periods,
                      link, design, rng, categories)
    # 断言夹具确实缺等级, 防止采样偶然破坏场景
    used_by_sparse = {r["raw_label"] for r in rows
                      if r["rater"] == "R2_sparse"}
    assert "一般" not in used_by_sparse, "缺等级夹具未能避免'一般'等级"
    return SyntheticDataset(
        rows, categories,
        {"link": link, "seed": seed,
         "abilities": dict(zip(projects, abilities.tolist())),
         "thresholds": {r: thresholds[i].tolist()
                        for i, r in enumerate(raters)},
         "sparse_rater": "R2_sparse",
         "missing_category": "一般",
         "note": "R2_sparse 阈值间隙过宽, 等级'一般'确定性缺失",
         "periods": periods},
        scenario="missing")


def make_tiny(link: str = "logit") -> SyntheticDataset:
    """确定性的极小夹具: 3 项目 x 2 评价者 x 1 期, 3 个等级, 无随机抽样。

    行内直接给出概率, 供 tests/test_likelihood.py 手工核对。
    """
    categories = ["很差", "一般", "满意"]
    projects = ["P0", "P1", "P2"]
    raters = ["R0", "R1"]
    periods = [0]
    abilities = np.array([-1.0, 0.0, 1.0])
    thresholds = np.array([
        [[-0.5, 1.0]],   # R0
        [[0.2, 1.6]],    # R1
    ])
    F, _, _, _ = link_funs(link)
    rows = []
    for ri, r in enumerate(raters):
        for pi, p in enumerate(projects):
            z = np.clip(thresholds[ri, 0] - abilities[pi], -30, 30)
            cum = F(z)
            probs = np.array([cum[0], cum[1] - cum[0], 1 - cum[1]])
            for k, lab in enumerate(categories):
                rows.append({"project": p, "rater": r, "period": 0,
                             "raw_label": lab,
                             "true_probability": float(probs[k])})
    return SyntheticDataset(
        rows, categories,
        {"link": link,
         "abilities": dict(zip(projects, abilities.tolist())),
         "thresholds": {r: thresholds[i].tolist()
                        for i, r in enumerate(raters)},
         "note": "每行是一个 (项目,评价者,等级) 概率单元, 非抽样观测",
         "periods": periods},
        scenario="tiny")


SCENARIOS = {
    "base": make_base,
    "reversal": make_reversal,
    "separation": make_separation,
    "missing": make_missing_category,
    "tiny": make_tiny,
}
