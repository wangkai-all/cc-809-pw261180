"""模型选择: 固定阈值 vs 缓变阈值。

两条纪律
--------
1. **整项目留出 (leave-one-project-out)**: 留出单位是项目的 *全部* 评价,
   绝不是随机抽单行 —— 单行留出会让同一项目的其他评价泄漏项目能力信息。
   留出项目在训练集中完全不存在, 用零和锚下的先验预测 (b=0) 评估,
   衡量的是阈值尺度对陌生项目的标定质量。
2. **滚动起源 (rolling origin)**: 用时期 <=t 的数据拟合, 只预测 t+1 的
   评价; 任何拟合都不借用未来评价校正过去。

汇总指标为平均每条留出评价的对数预测密度 (mean log predictive density,
越大越好), 同时给出 Brier 分数与正确分类率。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import stats
from .fit import fit_model
from .models import RatingRecord
from .stats import FitConfig, category_probs


@dataclass
class HoldoutResult:
    method: str
    config: dict[str, Any]
    fold: str
    n_held_out: int
    mean_log_pred_density: float
    brier: float
    accuracy: float
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _records_to_projection(records: list[RatingRecord], categories: list[str],
                           version_id: Optional[str] = None):
    from .store import Projection
    return Projection(categories=list(categories), records=list(records),
                      version_id=version_id, n_events=-1, n_corrections=0)


def _score(records: list[RatingRecord], categories: list[str],
           abilities: np.ndarray, thresholds: np.ndarray,
           train_index, link: str) -> list[dict[str, Any]]:
    """对一组记录计算预测指标; 未见过的项目能力取 0 (先验中心)。"""
    out = []
    if not records:
        return out
    pi = np.array([train_index.project_idx.get(r.project, -1)
                   for r in records])
    ri = np.array([train_index.rater_idx[r.rater] for r in records])
    ti = np.array([train_index.period_idx.get(r.period, -1)
                   for r in records])
    yi = np.array([train_index.label_idx[r.effective_label]
                   for r in records])
    # 未见项目: 用先验中心 b=0; 未见时期: 用最近已知时期
    safe_pi = np.where(pi >= 0, pi, 0)
    safe_ti = np.where(ti >= 0, ti, thresholds.shape[1] - 1)
    probs = category_probs(abilities, thresholds, safe_pi, ri, safe_ti, link)
    unseen = pi < 0
    if unseen.any():
        # b=0: 用零能力重算未见项目行
        zero_ability = np.zeros_like(abilities)
        probs[unseen] = category_probs(
            zero_ability, thresholds, np.zeros(unseen.sum(), dtype=int),
            ri[unseen], safe_ti[unseen], link)
    K = len(categories)
    onehot = np.eye(K)[yi]
    for n, r in enumerate(records):
        out.append({
            "event_id": r.event_id, "project": r.project, "rater": r.rater,
            "period": r.period, "observed": r.effective_label,
            "log_pred": float(np.log(np.clip(probs[n, yi[n]], 1e-300, None))),
            "brier": float(((probs[n] - onehot[n]) ** 2).sum()),
            "hit": bool(int(np.argmax(probs[n])) == yi[n]),
            "unseen_project": bool(unseen[n]),
        })
    return out


def _summarize(method: str, config: FitConfig, fold: str,
               details: list[dict[str, Any]]) -> HoldoutResult:
    if not details:
        return HoldoutResult(method, config.to_dict(), fold, 0,
                             float("nan"), float("nan"), float("nan"))
    return HoldoutResult(
        method, config.to_dict(), fold, len(details),
        mean_log_pred_density=float(np.mean([d["log_pred"] for d in details])),
        brier=float(np.mean([d["brier"] for d in details])),
        accuracy=float(np.mean([d["hit"] for d in details])),
        details=details)


def leave_one_project_out(proj, config: FitConfig
                          ) -> list[HoldoutResult]:
    """整项目留出交叉验证 (每折留出一个项目的全部评价)。"""
    results = []
    projects = proj.projects()
    for held in projects:
        train_recs = [r for r in proj.records if r.project != held]
        held_recs = [r for r in proj.records if r.project == held]
        if not train_recs or not held_recs:
            continue
        train_proj = _records_to_projection(
            train_recs, proj.categories, proj.version_id)
        fit = fit_model(train_proj, config, n_starts=2)
        train_index = train_proj.index()
        details = _score(held_recs, proj.categories, fit.abilities,
                         fit.thresholds, train_index, config.link)
        results.append(_summarize("leave_one_project_out", config,
                                  f"holdout={held}", details))
    return results


def rolling_origin(proj, config: FitConfig) -> list[HoldoutResult]:
    """用 <=t 期训练, 预测 t+1 期; 第一期没有历史, 从第二期开始。"""
    results = []
    periods = proj.periods()
    for t in periods[1:]:
        train_recs = [r for r in proj.records if r.period <= t - 1]
        held_recs = [r for r in proj.records if r.period == t]
        if not train_recs or not held_recs:
            continue
        train_proj = _records_to_projection(
            train_recs, proj.categories, proj.version_id)
        fit = fit_model(train_proj, config, n_starts=2)
        train_index = train_proj.index()
        details = _score(held_recs, proj.categories, fit.abilities,
                         fit.thresholds, train_index, config.link)
        results.append(_summarize("rolling_origin", config,
                                  f"train<=period{t - 1}->predict period{t}",
                                  details))
    return results


def compare_models(proj, link: str = "logit",
                   sigma_walk: float = 0.25) -> dict[str, Any]:
    """对 fixed / drift 两种配置执行全部验证并给出选择建议。"""
    configs = [
        FitConfig(kind="fixed", link=link),
        FitConfig(kind="drift", link=link, sigma_walk=sigma_walk),
    ]
    output: dict[str, Any] = {"link": link, "models": []}
    for cfg in configs:
        lopo = leave_one_project_out(proj, cfg)
        roll = rolling_origin(proj, cfg)
        all_details = [d for hr in lopo + roll for d in hr.details]
        output["models"].append({
            "config": cfg.to_dict(),
            "lopo": [hr.to_dict() for hr in lopo],
            "rolling_origin": [hr.to_dict() for hr in roll],
            "aggregate": _summarize("combined", cfg, "all",
                                    all_details).to_dict(),
        })
    a, b = output["models"]
    # 以滚动起源 MLPD 为主 (直接检验时间预测), LOPO 为辅
    roll_a = np.mean([h["mean_log_pred_density"]
                      for h in a["rolling_origin"]
                      if h["n_held_out"] > 0])
    roll_b = np.mean([h["mean_log_pred_density"]
                      for h in b["rolling_origin"]
                      if h["n_held_out"] > 0])
    chosen = "drift" if roll_b > roll_a + 1e-4 else "fixed"
    output["selection"] = {
        "criterion": "rolling_origin mean log predictive density "
                     "(无未来信息泄漏)",
        "fixed_mlpd": float(roll_a),
        "drift_mlpd": float(roll_b),
        "selected": chosen,
        "rationale": ("drift 的时间外推 MLPD 更高, 阈值存在时期漂移"
                      if chosen == "drift"
                      else "fixed 的时间外推 MLPD 不劣于 drift, "
                           "按简约原则选择固定阈值"),
    }
    return output
