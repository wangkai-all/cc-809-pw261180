"""只读网页与 JSON API (标准库 http.server, 无第三方依赖)。

页面:
  /                     总览 (版本、问题清单、项目能力)
  /project.html?id=...  同一项目在不同评价者眼中的等级概率 + 锚定调整
  /thresholds.html      各评价者阈值, 固定 vs 漂移对比
  /data.html            评价明细 (原文本/生效文本/更正标记) 与版本链
  /reports.html         报告清单与复现状态

所有图表为原生 SVG, 配色使用经校验的分类色板, 并始终提供表格视图。
"""
from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import diagnostics, report
from .fit import fit_model
from .select import compare_models
from .stats import FitConfig, predict_proba
from .store import EventStore

WEB_DIR = Path(__file__).parent / "web_static"


class WebState:
    def __init__(self, store: EventStore, link: str = "logit",
                 sigma_walk: float = 0.25):
        self.store = store
        self.link = link
        self.sigma_walk = sigma_walk
        self._lock = threading.Lock()

    # ------------------------------------------------------------ 数据装配
    def overview(self) -> dict[str, Any]:
        proj = self.store.project()
        if not proj.records:
            return {"empty": True, "categories": [],
                    "versions": [v.to_dict() for v in self.store.versions()]}
        fixed = fit_model(proj, FitConfig(kind="fixed", link=self.link))
        drift = fit_model(proj, FitConfig(
            kind="drift", link=self.link, sigma_walk=self.sigma_walk))
        diag = diagnostics.diagnose(proj, fixed)
        return {
            "empty": False,
            "version_id": proj.version_id,
            "categories": proj.categories,
            "n_ratings": len(proj.records),
            "n_corrections": proj.n_corrections,
            "projects": proj.projects(),
            "raters": proj.raters(),
            "periods": proj.periods(),
            "abilities": fixed.ability_table(),
            "diagnostics": diag.to_dict(),
            "fit_stats": diag.fit_stats,
            "fixed_log_likelihood": fixed.log_likelihood,
            "drift_log_likelihood": drift.log_likelihood,
            "versions": [v.to_dict() for v in self.store.versions()],
            "reports": self.store.list_reports(),
        }

    def project_detail(self, project_id: str) -> dict[str, Any]:
        proj = self.store.project()
        fixed = fit_model(proj, FitConfig(kind="fixed", link=self.link))
        drift = fit_model(proj, FitConfig(
            kind="drift", link=self.link, sigma_walk=self.sigma_walk))
        idx = proj.index()
        if project_id not in idx.project_idx:
            raise KeyError(project_id)
        pi = idx.project_idx[project_id]
        rows = []
        for model_name, fit in (("fixed", fixed), ("drift", drift)):
            for ri, r in enumerate(fit.raters):
                for ti, t in enumerate(fit.periods):
                    probs = predict_proba(
                        fit.abilities, fit.thresholds,
                        pi, ri, ti, self.link)
                    rows.append({"model": model_name, "rater": r,
                                 "period": t,
                                 "probs": [float(x) for x in probs]})
        # 该项目的实际观测 (保留原文本)
        observed = [r.to_dict() for r in proj.records
                    if r.project == project_id]
        return {"project": project_id,
                "categories": proj.categories,
                "ability": float(fixed.abilities[pi]),
                "ability_se": float(fixed.se_abilities[pi]),
                "probability_rows": rows,
                "observed": observed}

    def thresholds_view(self) -> dict[str, Any]:
        proj = self.store.project()
        fixed = fit_model(proj, FitConfig(kind="fixed", link=self.link))
        drift = fit_model(proj, FitConfig(
            kind="drift", link=self.link, sigma_walk=self.sigma_walk))
        diag = diagnostics.diagnose(proj, drift)
        comparison = compare_models(
            proj, link=self.link, sigma_walk=self.sigma_walk)
        return {
            "categories": proj.categories,
            "raters": proj.raters(),
            "periods": proj.periods(),
            "fixed": {"threshold_table": fixed.threshold_table()},
            "drift": {"threshold_table": drift.threshold_table(),
                      "drift_summary": diag.drift_summary},
            "selection": comparison["selection"],
        }

    def data_view(self) -> dict[str, Any]:
        proj = self.store.project()
        return {"categories": proj.categories,
                "version_id": proj.version_id,
                "records": [r.to_dict() for r in proj.records],
                "events": [e.to_dict() for e in self.store.events],
                "versions": [v.to_dict() for v in self.store.versions()]}

    def reports_view(self) -> dict[str, Any]:
        out = []
        for meta in self.store.list_reports():
            rid = meta["report_id"]
            saved = self.store.load_report(rid)
            current_v = self.store.project().version_id
            out.append({
                **meta,
                "valid_now": saved["version_id"] == current_v,
                "validity": saved.get("validity"),
                "n_problems": len(
                    saved.get("diagnostics", {}).get("problems", [])),
                "selection": (saved.get("selection") or {}).get("selection"),
            })
        return {"reports": out}

    def reproduce(self, report_id: str) -> dict[str, Any]:
        return report.reproduce_report(self.store, report_id)

    def anchor_view(self) -> dict[str, Any]:
        """锚定调整: 返回零和锚结果, 前端可切换锚定项目 (纯展示重表达)。"""
        proj = self.store.project()
        fit = fit_model(proj, FitConfig(kind="fixed", link=self.link))
        return {"projects": fit.projects,
                "abilities": fit.ability_table(),
                "thresholds_by_rater": fit.threshold_table()}


def make_handler(state: WebState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "OrdinalApp/1.0"

        def log_message(self, *args):  # 静默
            pass

        def _send(self, code: int, body: bytes,
                  content_type: str = "application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(
                obj, ensure_ascii=False).encode("utf-8"))

        def do_GET(self):
            parsed = urlparse(self.path)
            path, qs = parsed.path, parse_qs(parsed.query)
            try:
                if path == "/api/overview":
                    return self._json(state.overview())
                if path == "/api/project":
                    pid = qs.get("id", [""])[0]
                    return self._json(state.project_detail(pid))
                if path == "/api/thresholds":
                    return self._json(state.thresholds_view())
                if path == "/api/data":
                    return self._json(state.data_view())
                if path == "/api/reports":
                    return self._json(state.reports_view())
                if path == "/api/anchor":
                    return self._json(state.anchor_view())
                if path == "/api/reproduce":
                    rid = qs.get("id", [""])[0]
                    return self._json(state.reproduce(rid))
                # 静态文件
                if path == "/":
                    path = "/index.html"
                static_path = (WEB_DIR / path.lstrip("/")).resolve()
                if not str(static_path).startswith(str(WEB_DIR.resolve())) \
                        or not static_path.is_file():
                    return self._send(404, b"not found", "text/plain")
                ctype = mimetypes.guess_type(str(static_path))[0] \
                    or "application/octet-stream"
                return self._send(200, static_path.read_bytes(), ctype)
            except KeyError as exc:
                return self._json({"error": f"未知标识: {exc}"}, 404)
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": str(exc)}, 500)

    return Handler


def serve(store: EventStore, host: str = "127.0.0.1", port: int = 8000,
          link: str = "logit", sigma_walk: float = 0.25) -> ThreadingHTTPServer:
    state = WebState(store, link=link, sigma_walk=sigma_walk)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    return httpd
