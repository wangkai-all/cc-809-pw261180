"""可识别性与拟合诊断。

核心问题
--------
1. **分离 (separation)**: 若两组项目 *从不共享任何评价者*, 它们在潜在尺度上
   的相对位置没有任何似然信息, 联合估计会把差异推向无穷 (MLE) 或由先验
   任意决定 (MAP)。通过 "项目-项目重叠图" 的连通分量识别, 并明确报告
   "尺度分离、无法跨分量比较"。
2. **评价者分离**: 某评价者只评过一个项目时, 其阈值与该项目能力互相混杂,
   阈值绝对值不可识别。
3. **缺等级**: 某评价者 (或全局) 从未使用过某个等级 -> 相邻阈值估计被推向
   无穷, 报告为不稳定。
4. **尺度反转**: 评价者的阈值整体排序与多数人相反 (倾向把好项目评得更差),
   用锚定项目上的平均等级相关/阈值符号检测。
5. **阈值漂移**: drift 模型下相邻期阈值位置的后验变化及其不确定性。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .fit import FitResult
from .stats import TensorData


# ------------------------------------------------------------ 重叠图连通分量
def overlap_components(proj) -> list[list[str]]:
    """项目-项目重叠图: 被同一评价者评过的两个项目之间连边。

    返回连通分量 (项目 id 列表, 按分量内最小 id 排序)。
    """
    raters_to_projects: dict[str, set[str]] = {}
    for r in proj.records:
        raters_to_projects.setdefault(r.rater, set()).add(r.project)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for ps in raters_to_projects.values():
        ps = list(ps)
        for q in ps[1:]:
            union(ps[0], q)
    groups: dict[str, list[str]] = {}
    for p in parent:
        groups.setdefault(find(p), []).append(p)
    comps = [sorted(v) for v in groups.values()]
    return sorted(comps, key=lambda c: c[0])


def rater_overlap_table(proj) -> list[dict[str, Any]]:
    """每个评价者评过哪些项目, 以及每对项目共享哪些评价者。"""
    r2p: dict[str, set[str]] = {}
    for r in proj.records:
        r2p.setdefault(r.rater, set()).add(r.project)
    return [{"rater": r, "projects": sorted(ps), "n_projects": len(ps)}
            for r, ps in sorted(r2p.items())]


# ------------------------------------------------------------ 等级使用情况
def category_usage(proj) -> dict[str, Any]:
    cats = proj.categories
    idx = proj.index()
    global_count = {c: 0 for c in cats}
    per_rater: dict[str, dict[str, int]] = {
        r: {c: 0 for c in cats} for r in idx.raters}
    for rec in proj.records:
        global_count[rec.effective_label] += 1
        per_rater[rec.rater][rec.effective_label] += 1
    missing_global = [c for c, n in global_count.items() if n == 0]
    missing_by_rater = {r: [c for c, n in d.items() if n == 0]
                        for r, d in per_rater.items()}
    return {
        "global_count": global_count,
        "per_rater": per_rater,
        "missing_global": missing_global,
        "missing_by_rater": missing_by_rater,
    }


# ------------------------------------------------------------ 尺度反转检测
def _rater_project_matrix(proj) -> tuple[np.ndarray, list[str]]:
    """评价者 x 项目 的平均等级下标矩阵 (缺失为 NaN)。"""
    idx = proj.index()
    pi = np.array([idx.project_idx[r.project] for r in proj.records])
    ri = np.array([idx.rater_idx[r.rater] for r in proj.records])
    yi = np.array([idx.label_idx[r.effective_label]
                   for r in proj.records])
    P, R = len(idx.projects), len(idx.raters)
    sums = np.full((R, P), 0.0)
    cnt = np.zeros((R, P))
    np.add.at(sums, (ri, pi), yi)
    np.add.at(cnt, (ri, pi), 1)
    M = np.divide(sums, cnt, out=np.full((R, P), np.nan), where=cnt > 0)
    return M, idx.raters


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """只用两评价者共同评过的项目计算 Kendall tau-b (并列相容)。"""
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan
    concord = discord = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            sx = np.sign(x[i] - x[j])
            sy = np.sign(y[i] - y[j])
            if sx == 0 or sy == 0:
                continue
            if sx == sy:
                concord += 1
            else:
                discord += 1
    denom = concord + discord
    return float((concord - discord) / denom) if denom else np.nan


def _pearson_common(x: np.ndarray, y: np.ndarray) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rater_orientation(proj) -> list[dict[str, Any]]:
    """每个评价者的 "方向": 与 *留一者共识* 在共同项目上的 Pearson 相关。

    做法: 检测评价者 r 时, 共识由其余评价者的项目平均等级下标构成,
    因此一个反转评价者不会污染它自己的参照。正常评价者与共识强正相关,
    尺度反转者 (把好项目系统性评差) 呈负相关 (阈值 <= -0.35)。
    等级下标仅用于方向检测, 不参与建模。
    """
    M, raters = _rater_project_matrix(proj)
    R = len(raters)
    out = []
    for r in range(R):
        others = [j for j in range(R) if j != r]
        if not others:
            out.append({"rater": raters[r],
                        "n_ratings": int((~np.isnan(M[r])).sum()),
                        "corr_with_leave_one_out_consensus": None,
                        "reversed": False,
                        "note": "只有一名评价者, 无法判断方向"})
            continue
        # 重叠门控: 只与 *评过足够共同项目* 的评价者建立共识, 避免把
        # 尺度分离或稀疏单点噪声误判为方向反转
        shared = [j for j in others
                  if int(np.sum(~np.isnan(M[r]) & ~np.isnan(M[j]))) >= 4]
        if not shared:
            out.append({"rater": raters[r],
                        "n_ratings": int((~np.isnan(M[r])).sum()),
                        "corr_with_leave_one_out_consensus": None,
                        "reversed": False,
                        "note": "与其他评价者共同项目不足 (尺度分离或数据稀疏), "
                                "无法判定方向"})
            continue
        # 单一同伴时共识不稳健, 不据此判定反转
        if len(shared) < 2:
            out.append({"rater": raters[r],
                        "n_ratings": int((~np.isnan(M[r])).sum()),
                        "n_shared_raters": len(shared),
                        "corr_with_leave_one_out_consensus": None,
                        "reversed": False,
                        "note": "仅与一名评价者有足够重叠, 参照不足, 不判定反转"})
            continue
        with np.errstate(invalid="ignore"):
            consensus = np.nanmean(M[shared], axis=0)
        # 共识在该评价者评过的项目上必须有有效值 (尺度分离防护)
        valid_consensus = ~np.isnan(consensus) & ~np.isnan(M[r])
        if int(np.sum(valid_consensus)) < 3:
            out.append({"rater": raters[r],
                        "n_ratings": int((~np.isnan(M[r])).sum()),
                        "corr_with_leave_one_out_consensus": None,
                        "reversed": False,
                        "note": "与其他评价者共同项目不足 (尺度分离), 无法判定方向"})
            continue
        rho = _pearson_common(M[r], consensus)
        if not np.isfinite(rho):
            out.append({"rater": raters[r],
                        "n_ratings": int((~np.isnan(M[r])).sum()),
                        "corr_with_leave_one_out_consensus": None,
                        "reversed": False,
                        "note": "共同项目不足或无变异, 无法判断方向"})
            continue
        out.append({
            "rater": raters[r],
            "n_ratings": int(np.sum(~np.isnan(M[r]))),
            "n_shared_raters": len(shared),
            "corr_with_leave_one_out_consensus": rho,
            "reversed": rho <= -0.35,
            "note": ("与留一者共识相关 %.2f <= -0.35, 疑似尺度反转" % rho)
                    if rho <= -0.35 else "",
        })
    return out


# ------------------------------------------------------------ 综合诊断
@dataclass
class DiagnosticReport:
    version_id: str
    n_observations: int
    project_components: list[list[str]]
    separated: bool
    separation_detail: str
    rater_single_project: list[str]
    missing_categories_global: list[str]
    missing_categories_by_rater: dict[str, list[str]]
    orientation: list[dict[str, Any]]
    drift_summary: list[dict[str, Any]]
    fit_warnings: list[str]
    unstable_thresholds: list[dict[str, Any]]
    fit_stats: dict[str, float]
    problems: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def diagnose(proj, fit: FitResult | None = None) -> DiagnosticReport:
    comps = overlap_components(proj)
    separated = len(comps) > 1
    if separated:
        detail = ("项目重叠图有 %d 个连通分量, 分量间没有共同评价者; "
                  "跨分量的项目能力差异无似然信息, MAP 结果仅反映先验, "
                  "不可跨分量比较。分量: %s"
                  % (len(comps), comps))
    else:
        detail = "全部项目通过共同评价者连通, 项目尺度可联合识别。"

    rtable = rater_overlap_table(proj)
    single = [row["rater"] for row in rtable if row["n_projects"] == 1]

    usage = category_usage(proj)
    orientation = rater_orientation(proj)

    problems: list[dict[str, str]] = []
    if separated:
        problems.append({"level": "error", "kind": "separation",
                         "message": detail})
    if single:
        problems.append({
            "level": "warning", "kind": "rater_separation",
            "message": ("评价者只评过单一项目, 阈值与项目能力混杂, "
                        "阈值绝对值不可识别: %s" % single)})
    if usage["missing_global"]:
        problems.append({
            "level": "warning", "kind": "missing_category",
            "message": "整个数据中从未出现的等级: %s, 对应阈值无信息"
                       % usage["missing_global"]})
    for r, miss in usage["missing_by_rater"].items():
        if miss:
            problems.append({
                "level": "warning", "kind": "missing_category_rater",
                "message": f"评价者 {r} 从未使用等级 {miss}"})
    for o in orientation:
        if o["reversed"]:
            problems.append({
                "level": "warning", "kind": "reversed_rater",
                "message": f"评价者 {o['rater']} 疑似尺度反转 "
                           "(与留一者共识相关="
                           f"{o['corr_with_leave_one_out_consensus']:.2f}, "
                           "阈值 -0.35)"})

    drift_summary = []
    unstable = []
    fit_warnings: list[str] = []
    fit_stats: dict[str, float] = {}
    if fit is not None:
        fit_warnings = list(fit.warnings)
        if not fit.hessian_ok:
            problems.append({"level": "error", "kind": "hessian",
                             "message": "Hessian 不可逆, 存在不可识别参数"})
        # 漂移: 每评价者相邻期阈值位置变化
        if fit.config.kind == "drift":
            loc = fit.thresholds[:, :, 0]  # (R,T) 第一条阈值即位置
            se_loc = fit.se_thresholds[:, :, 0]
            for ri, r in enumerate(fit.raters):
                for ti in range(1, len(fit.periods)):
                    delta = loc[ri, ti] - loc[ri, ti - 1]
                    se_delta = np.sqrt(
                        np.nan_to_num(se_loc[ri, ti]) ** 2
                        + np.nan_to_num(se_loc[ri, ti - 1]) ** 2)
                    z = delta / se_delta if se_delta > 0 else np.nan
                    drift_summary.append({
                        "rater": r,
                        "from_period": fit.periods[ti - 1],
                        "to_period": fit.periods[ti],
                        "delta_location": float(delta),
                        "se_delta": float(se_delta),
                        "z": float(z) if np.isfinite(z) else None,
                        "significant": bool(np.isfinite(z) and abs(z) >= 2),
                    })
        # 不稳定阈值: SE 过大或非有限
        for ri, r in enumerate(fit.raters):
            for ti, t in enumerate(fit.periods):
                for k in range(len(fit.categories) - 1):
                    se = fit.se_thresholds[ri, ti, k]
                    if not np.isfinite(se) or se > 2.0:
                        unstable.append({"rater": r, "period": t,
                                         "threshold_index": k, "se": float(se)})
        if unstable:
            problems.append({
                "level": "warning", "kind": "unstable_threshold",
                "message": "%d 个阈值标准误 > 2 或不可计算 (缺等级/分离)"
                           % len(unstable)})
        fit_stats = _fit_stats(proj, fit)
    return DiagnosticReport(
        version_id=proj.version_id or "unversioned",
        n_observations=len(proj.records),
        project_components=comps,
        separated=separated,
        separation_detail=detail,
        rater_single_project=single,
        missing_categories_global=usage["missing_global"],
        missing_categories_by_rater={
            r: m for r, m in usage["missing_by_rater"].items() if m},
        orientation=orientation,
        drift_summary=drift_summary,
        fit_warnings=fit_warnings,
        unstable_thresholds=unstable,
        fit_stats=fit_stats,
        problems=problems,
    )


def _fit_stats(proj, fit: FitResult) -> dict[str, float]:
    data = TensorData.from_projection(proj)
    ll = fit.log_likelihood
    n = fit.n_observations
    # 空模型: 全部观测共享一组边际等级概率
    marginals = np.bincount(data.y, minlength=len(proj.categories)) / n
    null_ll = float(np.sum(
        np.bincount(data.y, minlength=len(proj.categories))
        * np.log(np.clip(marginals, 1e-12, None))))
    k = fit.n_params
    return {
        "log_likelihood": ll,
        "null_log_likelihood": null_ll,
        "mcfadden_pseudo_r2": float(1.0 - ll / null_ll) if null_ll != 0 else 0.0,
        "aic": float(2 * k - 2 * ll),
        "bic": float(k * np.log(max(n, 1)) - 2 * ll),
        "n": float(n),
        "k": float(k),
    }
