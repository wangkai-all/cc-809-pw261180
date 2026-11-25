"""MAP 拟合与结果封装。

拟合在 *自由参数* 空间做 L-BFGS-B (多起点), 阈值由 softplus 间隙构造,
因此任何收敛点的阈值都严格有序, 不存在交叉。先验保证完全分离场景下
后验仍为正常分布 (MLE 会跑到无穷远), 此时标准误会很大并由诊断模块标记。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize

from . import stats
from .stats import (FitConfig, ParamLayout, TensorData, hessian_finite_diff,
                    link_funs, log_likelihood_natural, neg_log_posterior,
                    neg_log_posterior_grad)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class FitResult:
    fit_id: str
    version_id: str
    config: FitConfig
    categories: list[str]
    projects: list[str]
    raters: list[str]
    periods: list[int]
    abilities: np.ndarray
    thresholds: np.ndarray            # (R, T, K-1)
    se_abilities: np.ndarray
    se_thresholds: np.ndarray
    theta: np.ndarray
    log_posterior: float
    log_likelihood: float
    log_prior: float
    converged: bool
    optimizer_message: str
    n_observations: int
    n_params: int
    hessian_ok: bool
    warnings: list[str] = field(default_factory=list)

    def ability_table(self) -> list[dict[str, Any]]:
        return [{"project": p, "ability": float(self.abilities[i]),
                 "se": float(self.se_abilities[i])}
                for i, p in enumerate(self.projects)]

    def threshold_table(self) -> list[dict[str, Any]]:
        out = []
        K = len(self.categories)
        for ri, r in enumerate(self.raters):
            for ti, t in enumerate(self.periods):
                row = {"rater": r, "period": t}
                for k in range(K - 1):
                    row[f"tau_{k}"] = float(self.thresholds[ri, ti, k])
                    row[f"se_tau_{k}"] = float(self.se_thresholds[ri, ti, k])
                out.append(row)
        return out

    def probability_table(self, link: str) -> list[dict[str, Any]]:
        """每个 (项目, 评价者, 时期) 在各等级文本上的预测概率。"""
        F, _, _, _ = link_funs(link)
        rows = []
        for pi, p in enumerate(self.projects):
            for ri, r in enumerate(self.raters):
                for ti, t in enumerate(self.periods):
                    probs = stats.predict_proba(
                        self.abilities, self.thresholds, pi, ri, ti, link)
                    row = {"project": p, "rater": r, "period": t}
                    for k, lab in enumerate(self.categories):
                        row[f"P({lab})"] = float(probs[k])
                    row["expected_category_index"] = float(
                        np.sum(probs * np.arange(len(probs))))
                    rows.append(row)
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit_id": self.fit_id,
            "version_id": self.version_id,
            "created_ts": _utc(),
            "status": "ok" if self.converged else "failed",
            "config": self.config.to_dict(),
            "categories": self.categories,
            "projects": self.projects,
            "raters": self.raters,
            "periods": self.periods,
            "n_observations": self.n_observations,
            "n_params": self.n_params,
            "log_posterior": self.log_posterior,
            "log_likelihood": self.log_likelihood,
            "log_prior": self.log_prior,
            "converged": self.converged,
            "optimizer_message": self.optimizer_message,
            "hessian_ok": self.hessian_ok,
            "warnings": self.warnings,
            "ability_table": self.ability_table(),
            "threshold_table": self.threshold_table(),
            "probability_table": self.probability_table(self.config.link),
            "theta": self.theta.tolist(),
        }


def _natural_jacobian(layout: ParamLayout, theta: np.ndarray,
                      eps: float = 1e-5):
    """自由参数 -> 自然参数的数值雅可比 (中心差分)。

    返回:
      J_abilities  形状 (P, n_params)
      J_thresholds 形状 (R, T, K-1, n_params)
    """
    base_b, base_tau = layout.unpack(theta)
    Jb = np.zeros((layout.P, layout.n_params))
    Jt = np.zeros((layout.R, layout.T,
                   max(layout.K - 1, 1), layout.n_params))
    for j in range(layout.n_params):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        bp, taup = layout.unpack(tp)
        bm, taum = layout.unpack(tm)
        Jb[:, j] = (bp - bm) / (2 * eps)
        Jt[..., j] = (taup - taum) / (2 * eps)
    return Jb, Jt


def fit_model(proj, config: Optional[FitConfig] = None,
              *, fit_id: Optional[str] = None,
              n_starts: int = 3,
              verbose: bool = False) -> FitResult:
    """对一个投影执行 MAP 拟合。"""
    config = config or FitConfig()
    if config.kind not in ("fixed", "drift"):
        raise ValueError("config.kind 必须是 fixed 或 drift")
    data = TensorData.from_projection(proj)
    idx = proj.index()
    layout = ParamLayout(len(idx.projects), len(idx.raters),
                         len(idx.periods), len(idx.categories), config)
    rng = np.random.default_rng(42)
    best = None
    starts = [layout.initial_theta()]
    for _ in range(max(n_starts - 1, 0)):
        starts.append(starts[0] + 0.5 * rng.standard_normal(layout.n_params))

    t0 = time.time()
    for si, theta0 in enumerate(starts):
        try:
            opt = minimize(
                neg_log_posterior, theta0,
                args=(layout, data, config.link),
                jac=neg_log_posterior_grad, method="L-BFGS-B",
                options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"start {si} 异常: {exc}")
            continue
        if best is None or opt.fun < best.fun:
            best = opt
    if best is None:
        raise RuntimeError("所有优化起点均失败")

    theta_hat = best.x
    abilities, thresholds = layout.unpack(theta_hat)
    ll = log_likelihood_natural(abilities, thresholds, data, config.link)
    lp = layout.log_prior(theta_hat)
    warnings: list[str] = []
    hessian_ok = True
    se_abilities = np.full(len(idx.projects), np.nan)
    se_thresholds = np.full((len(idx.raters), len(idx.periods),
                             len(idx.categories) - 1), np.nan)
    try:
        H = hessian_finite_diff(theta_hat, layout, data, config.link)
        cov = np.linalg.inv(H)
        # 完整 delta 法: 数值构造 自由参数 -> 自然参数(能力, 全部阈值) 的
        # 雅可比 J, 自然参数协方差 = J Cov J^T。统一处理零和锚、softplus
        # 间隙与 drift 游走, 避免手工链式法则遗漏交叉项。
        J_abilities, J_thresholds = _natural_jacobian(layout, theta_hat)
        cov_ability = J_abilities @ cov @ J_abilities.T
        se_abilities = np.sqrt(np.clip(np.diag(cov_ability), 0, None))
        n_tau = layout.R * layout.T * (layout.K - 1)
        Jt = J_thresholds.reshape(n_tau, layout.n_params)
        cov_tau = Jt @ cov @ Jt.T
        se_thresholds = np.sqrt(
            np.clip(np.diag(cov_tau), 0, None)
        ).reshape(layout.R, layout.T, layout.K - 1)
    except np.linalg.LinAlgError:
        hessian_ok = False
        warnings.append("Hessian 不可逆: 参数不可识别 (常见于完全分离)")
    if not best.success:
        warnings.append(f"优化器未报告收敛: {best.message}")

    fit_id = fit_id or f"fit-{version_ish(proj)}-{config.kind}-{config.link}"
    return FitResult(
        fit_id=fit_id,
        version_id=proj.version_id or "unversioned",
        config=config,
        categories=list(idx.categories),
        projects=list(idx.projects),
        raters=list(idx.raters),
        periods=list(idx.periods),
        abilities=abilities,
        thresholds=thresholds,
        se_abilities=se_abilities,
        se_thresholds=se_thresholds,
        theta=theta_hat,
        log_posterior=float(-best.fun),
        log_likelihood=ll,
        log_prior=lp,
        converged=bool(best.success),
        optimizer_message=str(best.message),
        n_observations=len(data.y),
        n_params=layout.n_params,
        hessian_ok=hessian_ok,
        warnings=warnings,
    )


def version_ish(proj) -> str:
    return (proj.version_id or "x")[:10]
