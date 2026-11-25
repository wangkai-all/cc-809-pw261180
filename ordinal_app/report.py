"""报告: 把数据版本、拟合、诊断与模型选择固化为不可变快照。

更正事件后旧报告 *不* 自动失效: 报告内嵌版本哈希与数据快照,
随时可从事件日志按 n_events 重新投影并复现。同时显式标注报告相对于
当前日志是否仍然有效 (valid), 失效原因可追溯到具体更正事件。
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional

from . import diagnostics
from .fit import fit_model
from .select import compare_models
from .store import EventStore


def build_report(store: EventStore, *, kind: str = "fixed",
                 link: str = "logit", sigma_walk: float = 0.25,
                 include_selection: bool = True,
                 n_events: Optional[int] = None) -> dict[str, Any]:
    """在指定版本上生成完整报告 (默认当前版本)。"""
    proj = store.project(n_events=n_events)
    if not proj.records:
        raise ValueError("当前版本没有评价数据, 无法生成报告")
    from .stats import FitConfig
    cfg = FitConfig(kind=kind, link=link,
                    **({"sigma_walk": sigma_walk} if kind == "drift" else {}))
    fit = fit_model(proj, cfg, fit_id=(
        f"rptfit-{proj.version_id[:10]}-{kind}-{link}"))
    diag = diagnostics.diagnose(proj, fit)
    selection = compare_models(proj, link=link,
                               sigma_walk=sigma_walk) if include_selection \
        else None
    # 数据快照 (含原文本) 直接嵌入, 保证自包含复现
    snapshot = proj.to_dict()
    payload = {
        "report_id": "report-" + hashlib.sha256(
            (proj.version_id + kind + link).encode()).hexdigest()[:10],
        "version_id": proj.version_id,
        "n_events": proj.n_events,
        "fit_id": fit.fit_id,
        "created_from_latest": n_events is None,
        "fit": fit.to_dict(),
        "diagnostics": diag.to_dict(),
        "selection": selection,
        "data_snapshot": snapshot,
    }
    payload["valid"], payload["validity"] = _validity(store, proj)
    store.save_fit(fit.fit_id, payload["fit"])
    store.save_report(payload["report_id"], payload)
    return payload


def _validity(store: EventStore, proj) -> tuple[bool, dict[str, Any]]:
    """报告版本与日志当前状态的关系。"""
    latest = store.project()
    if latest.version_id == proj.version_id:
        return True, {"status": "current",
                      "message": "报告基于当前最新数据版本"}
    later_corr = [e.to_dict() for e in store.events[proj.n_events:]
                  if e.type.value == "correction"]
    later_rating = sum(1 for e in store.events[proj.n_events:]
                       if e.type.value == "rating")
    return False, {
        "status": "superseded",
        "message": "报告生成后出现新事件, 其中更正 %d 条、新评价 %d 条; "
                   "旧报告仍可按版本哈希复现, 但不再代表当前数据"
                   % (len(later_corr), later_rating),
        "later_corrections": later_corr,
    }


def reproduce_report(store: EventStore, report_id: str) -> dict[str, Any]:
    """从事件日志重放报告所用版本, 重新拟合并核对可复现性。"""
    saved = store.load_report(report_id)
    n_events = saved["n_events"]
    proj = store.project(n_events=n_events,
                         version_id=saved["version_id"])
    cfg_d = saved["fit"]["config"]
    from .stats import FitConfig
    cfg = FitConfig(**cfg_d)
    refit = fit_model(proj, cfg)
    saved_fit = saved["fit"]
    max_diff = max(
        max(abs(a - b) for a, b in zip(refit.abilities,
                                       [r["ability"]
                                        for r in saved_fit["ability_table"]])),
        max(abs(a - b)
            for a, b in zip(refit.thresholds.ravel(),
                            np_ravel_thresholds(saved_fit))),
    )
    return {
        "report_id": report_id,
        "version_id": saved["version_id"],
        "reproduced": bool(max_diff < 1e-4),
        "max_parameter_difference": float(max_diff),
        "log_likelihood_saved": saved_fit["log_likelihood"],
        "log_likelihood_refit": refit.log_likelihood,
        "valid_now": store.project().version_id == saved["version_id"],
        "validity_at_build": saved.get("validity"),
    }


def np_ravel_thresholds(fit_dict: dict[str, Any]):
    import numpy as np
    K = len(fit_dict["categories"])
    rows = fit_dict["threshold_table"]
    arr = np.array([[r[f"tau_{k}"] for k in range(K - 1)] for r in rows])
    return arr.ravel()
