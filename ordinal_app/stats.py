"""累积链接有序模型 (cumulative link model / graded-response) 的概率与似然。

模型
----
对项目 p 的潜在表现记为 b_p (越大越好), 评价者 r 在时期 t 持有一组
*严格有序* 的阈值 tau_{r,t,0} < tau_{r,t,1} < ... < tau_{r,t,K-2},
观测到等级 k 的概率为::

    P(Y <= k | p, r, t) = F(tau_{r,t,k} - b_p)
    P(Y = k)            = F(tau_k - b_p) - F(tau_{k-1} - b_p)

其中 F 为 logistic 或标准正态 CDF。两端用 -inf / +inf 处理。

* 等级只作为 *有序类别* 进入模型, 绝不按等间距数字平均。
* 阈值用 "位置 + softplus 间隙" 参数化, 从构造上保证不交叉
  (间隙恒正), 优化在无约束空间进行。
* 可识别性: 项目能力以零和为锚 (b_0 = -sum_{j>0} b_j);
  全部参数带弱信息先验, 完全分离时后验仍正常 (MLE 会发散)。
* 缓变阈值: a_{r,t} = a_{r,0} + sum_{s<=t} w_{r,s}, 游走增量
  w ~ N(0, sigma_walk); sigma_walk 越大越接近自由阈值, 越大惩罚越强
  越接近固定阈值。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, log_ndtr
from scipy.stats import norm

# ---------------------------------------------------------------- 链接函数
INF = 30.0  # 截断的无限远, log(Phi(-30)) 仍远小于数据量级


def link_funs(link: str):
    """返回 (F, logF, log1mF, f密度) 四元组。"""
    if link == "logit":
        def F(x):
            return expit(x)

        def logF(x):
            return -np.logaddexp(0.0, -x)

        def log1mF(x):
            return -np.logaddexp(0.0, x)

        def f(x):
            p = expit(x)
            return p * (1.0 - p)
        return F, logF, log1mF, f
    if link == "probit":
        F = norm.cdf

        def logF(x):
            return log_ndtr(x)

        def log1mF(x):
            return log_ndtr(-x)

        f = norm.pdf
        return F, logF, log1mF, f
    raise ValueError(f"未知链接: {link!r} (支持 logit / probit)")


# ---------------------------------------------------------------- 参数布局
@dataclass
class FitConfig:
    kind: str = "fixed"            # fixed | drift
    link: str = "logit"
    sigma_b: float = 1.5           # 项目能力先验尺度
    sigma_a: float = 1.5           # 阈值位置先验尺度
    sigma_g: float = 1.0           # 间隙原始参数先验尺度
    sigma_walk: float = 0.25       # 游走增量先验尺度 (drift 模型)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class ParamLayout:
    """自由参数 <-> 自然参数 (能力, 阈值) 的映射。

    自由参数顺序:
      fixed: [b_1..b_{P-1}, 每个rater: a_r, g_{r,1..K-2}]
      drift: [b_1..b_{P-1}, 每个rater: a_{r,0}, w_{r,1..T-1},
              g_{r,1..K-2}]
    """

    def __init__(self, n_projects: int, n_raters: int, n_periods: int,
                 n_categories: int, config: FitConfig):
        self.P, self.R, self.T, self.K = (
            n_projects, n_raters, n_periods, n_categories)
        self.cfg = config
        self.nb = max(self.P - 1, 0)
        self.ng = max(self.K - 2, 0)
        self.nw = (self.T - 1) if config.kind == "drift" else 0
        # 每个 rater 块大小
        self.block = 1 + self.nw + self.ng
        self.n_params = self.nb + self.R * self.block
        # 切片
        self.b_sl = slice(0, self.nb)
        self.rater_start = self.nb

    def rater_slice(self, r: int) -> slice:
        s = self.rater_start + r * self.block
        return slice(s, s + self.block)

    # ---------------------------------------------------------- 自由 -> 自然
    def unpack(self, theta: np.ndarray):
        """返回 abilities(P,), thresholds(R,T,K-1)。"""
        theta = np.asarray(theta, dtype=float)
        # 零和锚: ability_0 = -sum(b), 其余直接取
        bfree = theta[self.b_sl]
        abilities = np.empty(self.P)
        if self.P > 1:
            abilities[0] = -bfree.sum()
            abilities[1:] = bfree
        else:
            abilities[:] = 0.0
        thresholds = np.empty((self.R, self.T, max(self.K - 1, 1)))
        for r in range(self.R):
            blk = theta[self.rater_slice(r)]
            a0 = blk[0]
            if self.cfg.kind == "drift":
                w = blk[1:1 + self.nw]
                g = blk[1 + self.nw:]
                loc = np.concatenate([[a0], a0 + np.cumsum(w)])
            else:
                g = blk[1:]
                loc = np.full(self.T, a0)
            gaps = np.log1p(np.exp(np.clip(g, -INF, INF)))  # softplus
            # tau[t, k] = loc[t] + cumsum(gaps); k=0 时间隙和为 0
            cum = np.concatenate([[0.0], np.cumsum(gaps)])
            thresholds[r] = loc[:, None] + cum[None, :]
        return abilities, thresholds

    def initial_theta(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        theta = np.zeros(self.n_params)
        # 间隙初值取正: softplus(g)=0.8 -> g=log(exp(0.8)-1)
        g0 = np.log(np.exp(0.8) - 1.0)
        for r in range(self.R):
            blk = self.rater_slice(r)
            if self.cfg.kind == "drift":
                theta[blk][1 + self.nw:] = g0
            else:
                theta[blk][1:] = g0
        theta += 0.05 * rng.standard_normal(self.n_params)
        return theta

    # ---------------------------------------------------------- 先验 (自由空间)
    def log_prior(self, theta: np.ndarray) -> float:
        lp = 0.0
        cfg = self.cfg
        bfree = theta[self.b_sl]
        lp -= 0.5 * np.sum((bfree / cfg.sigma_b) ** 2)
        for r in range(self.R):
            blk = theta[self.rater_slice(r)]
            a0 = blk[0]
            lp -= 0.5 * (a0 / cfg.sigma_a) ** 2
            if cfg.kind == "drift":
                w = blk[1:1 + self.nw]
                g = blk[1 + self.nw:]
                lp -= 0.5 * np.sum((w / cfg.sigma_walk) ** 2)
            else:
                g = blk[1:]
            lp -= 0.5 * np.sum((g / cfg.sigma_g) ** 2)
        return float(lp)

    def grad_log_prior(self, theta: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(theta)
        cfg = self.cfg
        grad[self.b_sl] -= theta[self.b_sl] / cfg.sigma_b ** 2
        for r in range(self.R):
            blk = self.rater_slice(r)
            grad[blk][0] -= theta[blk][0] / cfg.sigma_a ** 2
            if cfg.kind == "drift":
                w_sl = slice(blk.start + 1, blk.start + 1 + self.nw)
                g_sl = slice(blk.start + 1 + self.nw, blk.stop)
                grad[w_sl] -= theta[w_sl] / cfg.sigma_walk ** 2
            else:
                g_sl = slice(blk.start + 1, blk.stop)
            grad[g_sl] -= theta[g_sl] / cfg.sigma_g ** 2
        return grad


# ---------------------------------------------------------------- 数据打包
@dataclass
class TensorData:
    """评价记录的张量化表示; 下标全部为整数, 文本等级由调用方保留。"""
    project_ix: np.ndarray   # (N,)
    rater_ix: np.ndarray     # (N,)
    period_ix: np.ndarray    # (N,)
    y: np.ndarray            # (N,) 等级下标 0..K-1
    categories: list[str]

    @classmethod
    def from_projection(cls, proj) -> "TensorData":
        idx = proj.index()
        if not proj.records:
            raise ValueError("投影中没有有效评价, 无法拟合")
        pi = np.array([idx.project_idx[r.project] for r in proj.records])
        ri = np.array([idx.rater_idx[r.rater] for r in proj.records])
        ti = np.array([idx.period_idx[r.period] for r in proj.records])
        yi = np.array([idx.label_idx[r.effective_label]
                       for r in proj.records])
        return cls(pi, ri, ti, yi, list(proj.categories))


# ---------------------------------------------------------------- 概率与似然
def category_probs(abilities: np.ndarray, thresholds: np.ndarray,
                   project_ix, rater_ix, period_ix, link: str):
    """每条记录在 K 个等级上的概率 (N, K)。"""
    F, _, _, _ = link_funs(link)
    K = thresholds.shape[-1] + 1
    b = abilities[project_ix]                                   # (N,)
    tau = thresholds[rater_ix, period_ix]                       # (N, K-1)
    z = np.clip(tau - b[:, None], -INF, INF)
    cum = np.empty((len(b), K - 1))
    cum = F(z)
    probs = np.empty((len(b), K))
    probs[:, 0] = cum[:, 0]
    probs[:, 1:K - 1] = cum[:, 1:] - cum[:, :-1]
    probs[:, K - 1] = 1.0 - cum[:, K - 2]
    # 数值清洗
    probs = np.clip(probs, 1e-300, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def log_likelihood_natural(abilities: np.ndarray, thresholds: np.ndarray,
                           data: TensorData, link: str) -> float:
    """自然参数 (能力、阈值) 下的对数似然, 供外部核对使用。"""
    probs = category_probs(abilities, thresholds, data.project_ix,
                           data.rater_ix, data.period_ix, link)
    return float(np.log(probs[np.arange(len(data.y)), data.y]).sum())


def predict_proba(abilities: np.ndarray, thresholds: np.ndarray,
                  project_ix: int, rater_ix: int, period_ix: int,
                  link: str) -> np.ndarray:
    """单条 (项目, 评价者, 时期) 的等级概率向量。"""
    return category_probs(
        abilities, thresholds,
        np.array([project_ix]), np.array([rater_ix]),
        np.array([period_ix]), link)[0]


# ---------------------------------------------------------------- 目标+梯度
def _nat_grad(theta, layout: ParamLayout, data: TensorData, link: str):
    """计算对数似然及其对自然参数 (能力, 阈值) 的梯度。"""
    abilities, thresholds = layout.unpack(theta)
    F, logF, log1mF, f = link_funs(link)
    N = len(data.y)
    b = abilities[data.project_ix]
    tau = thresholds[data.rater_ix, data.period_ix]          # (N, K-1)
    K = tau.shape[1] + 1
    z = np.clip(tau - b[:, None], -INF, INF)
    cum = F(z)

    # ---- 对数似然 (按等级位置分三类) ----
    logp = np.empty(N)
    edge0 = data.y == 0
    edgeK = data.y == K - 1
    interior = ~(edge0 | edgeK)
    logp[edge0] = logF(z[edge0, 0])
    logp[edgeK] = log1mF(z[edgeK, K - 2])
    if interior.any():
        yi = data.y[interior]
        diff = cum[interior, yi] - cum[interior, yi - 1]
        logp[interior] = np.log(np.clip(diff, 1e-300, None))
    ll = float(logp.sum())

    # ---- 自然参数梯度 ----
    g_ability = np.zeros(layout.P)
    g_tau = np.zeros((layout.R, layout.T, K - 1))
    dens = f(z)                                              # (N, K-1)

    # 最低等级 y=0: d logP / d z_0 = f(z0)/F(z0)
    if edge0.any():
        ii = np.where(edge0)[0]
        q = dens[ii, 0] / np.clip(cum[ii, 0], 1e-300, None)
        np.add.at(g_tau, (data.rater_ix[ii], data.period_ix[ii], 0), q)
        np.add.at(g_ability, data.project_ix[ii], -q)
    # 最高等级 y=K-1
    if edgeK.any():
        ii = np.where(edgeK)[0]
        kk = K - 2
        q = -dens[ii, kk] / np.clip(1.0 - cum[ii, kk], 1e-300, None)
        np.add.at(g_tau, (data.rater_ix[ii], data.period_ix[ii], kk), q)
        np.add.at(g_ability, data.project_ix[ii], -q)
    if interior.any():
        ii = np.where(interior)[0]
        yi = data.y[interior]
        diff = np.clip(cum[ii, yi] - cum[ii, yi - 1], 1e-300, None)
        q_hi = dens[ii, yi] / diff
        q_lo = -dens[ii, yi - 1] / diff
        np.add.at(g_tau, (data.rater_ix[ii], data.period_ix[ii], yi), q_hi)
        np.add.at(g_tau, (data.rater_ix[ii], data.period_ix[ii], yi - 1), q_lo)
        np.add.at(g_ability, data.project_ix[ii], -(q_hi + q_lo))
    return ll, abilities, thresholds, g_ability, g_tau


def _chain_rule(theta, layout: ParamLayout, g_ability, g_tau):
    """自然参数梯度 -> 自由参数梯度。"""
    grad = np.zeros(layout.n_params)
    # 能力: ability_0 = -sum(b_j), ability_{j+1} = b_j
    if layout.P > 1:
        grad[layout.b_sl] = g_ability[1:] - g_ability[0]
    for r in range(layout.R):
        blk = layout.rater_slice(r)
        # 位置 a_{r,t}: 每个阈值对当期位置导数为 1
        g_loc = g_tau[r].sum(axis=1)                        # (T,)
        grad[blk][0] += g_loc.sum()                         # a_0 影响所有 t
        if layout.cfg.kind == "drift":
            # a_t = a_0 + cumsum(w); w_s 影响所有 t>=s 的位置,
            # 故 dL/dw_s = sum_{t>=s} g_loc[t]
            tail = np.cumsum(g_loc[::-1])[::-1]   # tail[s] = sum_{t>=s}
            grad[blk][1:1 + layout.nw] += tail[1:]
            g_start = blk.start + 1 + layout.nw
        else:
            g_start = blk.start + 1
        # 间隙: tau_k = loc + sum_{m<=k} softplus(g_m), k 从 0 起;
        # g_m (m=1..K-2) 影响阈值 k>=m
        if layout.ng > 0:
            g_gap_nat = g_tau[r].sum(axis=0)                # (K-1,) 按 k
            # 对 g_m 的梯度 = sigmoid(g_m) * sum_{k>=m} g_gap_nat[k]
            gblk = theta[blk]
            g_vals = gblk[1 + layout.nw:] if layout.cfg.kind == "drift" \
                else gblk[1:]
            sig = expit(g_vals)
            tail = np.cumsum(g_gap_nat[::-1])[::-1]         # tail[m-1]=sum k>=m
            grad[g_start:g_start + layout.ng] += sig * tail[1:]
    return grad


def neg_log_posterior(theta, layout: ParamLayout, data: TensorData,
                      link: str):
    ll, _, _, g_ability, g_tau = _nat_grad(theta, layout, data, link)
    lp = layout.log_prior(theta)
    return -(ll + lp)


def neg_log_posterior_grad(theta, layout: ParamLayout, data: TensorData,
                           link: str):
    ll, _, _, g_ability, g_tau = _nat_grad(theta, layout, data, link)
    grad = _chain_rule(theta, layout, g_ability, g_tau)
    grad += layout.grad_log_prior(theta)
    return -grad


def numerical_gradient(theta, layout: ParamLayout, data: TensorData,
                       link: str, eps: float = 1e-6) -> np.ndarray:
    """有限差分数值梯度, 仅用于测试核对解析梯度。"""
    from scipy.optimize import approx_fprime
    return approx_fprime(
        theta, lambda t: neg_log_posterior(t, layout, data, link),
        eps).ravel()


def hessian_finite_diff(theta, layout: ParamLayout, data: TensorData,
                        link: str, eps: float = 1e-4) -> np.ndarray:
    """对 *解析梯度* 做一次有限差分得到负后验 Hessian。"""
    p = len(theta)
    H = np.empty((p, p))
    g0 = neg_log_posterior_grad(theta, layout, data, link)
    for j in range(p):
        t2 = theta.copy()
        t2[j] += eps
        g1 = neg_log_posterior_grad(t2, layout, data, link)
        H[:, j] = (g1 - g0) / eps
    return 0.5 * (H + H.T)
