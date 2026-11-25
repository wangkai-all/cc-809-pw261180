"""Append-only 事件存储、版本投影与拟合产物持久化。

目录结构 (应用根目录, 由使用者指定)::

    data/events.jsonl          只追加事件日志 (事实来源)
    data/versions.jsonl        每个版本的元数据 (哈希派生, 可重建)
    data/fits/<fit_id>.json    拟合产物 (引用版本哈希, 不可变)
    data/reports/<report_id>.json  报告快照 (含生成时全部输入)
    data/imports/              原始导入文件的副本 (按批次隔离)

所有写操作只追加; 不提供删除/修改事件的接口。
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import DataVersion, Event, EventType, RatingRecord, hash_events


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StoreError(Exception):
    pass


class EventStore:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.events_path = self.data_dir / "events.jsonl"
        self.versions_path = self.data_dir / "versions.jsonl"
        self.fits_dir = self.data_dir / "fits"
        self.reports_dir = self.data_dir / "reports"
        self.imports_dir = self.data_dir / "imports"
        for d in (self.data_dir, self.fits_dir, self.reports_dir, self.imports_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._events: list[Event] = self._read_events()
        self._rebuild_versions_file()

    # ------------------------------------------------------------------ 读取
    def _read_events(self) -> list[Event]:
        events: list[Event] = []
        if not self.events_path.exists():
            return events
        with self.events_path.open("r", encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(Event.from_dict(json.loads(line)))
                except Exception as exc:  # noqa: BLE001
                    raise StoreError(f"事件日志第 {ln} 行损坏: {exc}") from exc
        return events

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    # ------------------------------------------------------------------ 写入
    def append(self, event: Event) -> Event:
        """校验并追加事件; 返回同一个事件。"""
        with self._lock:
            self._validate_new(event)
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self._events.append(event)
            self._rebuild_versions_file()
        return event

    def _validate_new(self, event: Event) -> None:
        p = event.payload
        if event.type is EventType.IMPORT_BATCH:
            if any(e.payload.get("batch_id") == p["batch_id"]
                   for e in self._events if e.type is EventType.IMPORT_BATCH):
                raise StoreError(f"batch_id 已存在: {p['batch_id']}")
            # 同一库内等级文本集合必须一致 (有序等级体系不变)
            existing = self.categories()
            if existing is not None and existing != p["categories"]:
                raise StoreError(
                    f"等级体系冲突: 已有 {existing}, 新批次 {p['categories']}")
        elif event.type is EventType.RATING:
            if any(e.event_id == p["event_id"] for e in self._events):
                raise StoreError(f"event_id 重复: {p['event_id']}")
            cats = self.categories()
            if cats is not None and p["raw_label"] not in cats:
                raise StoreError(
                    f"等级文本 {p['raw_label']!r} 不在登记的等级体系 {cats} 中")
        elif event.type is EventType.CORRECTION:
            if any(e.event_id == p["event_id"] for e in self._events):
                raise StoreError(f"correction event_id 重复: {p['event_id']}")
            target = next((e for e in self._events
                           if e.type is EventType.RATING
                           and e.event_id == p["target_event_id"]), None)
            if target is None:
                raise StoreError(
                    f"更正目标不存在: {p['target_event_id']}")
            cats = self.categories()
            if cats is not None and p["new_raw_label"] not in cats:
                raise StoreError(
                    f"更正等级 {p['new_raw_label']!r} 不在等级体系 {cats} 中")
            if p["new_raw_label"] == self._effective_label(target.event_id):
                raise StoreError("更正后的等级与当前生效等级相同, 无需更正")

    def _effective_label(self, rating_event_id: str) -> Optional[str]:
        """事件日志当前状态下某 rating 的生效等级。"""
        label: Optional[str] = None
        for e in self._events:
            if e.type is EventType.RATING and e.event_id == rating_event_id:
                label = e.payload["raw_label"]
            elif (e.type is EventType.CORRECTION
                  and e.payload["target_event_id"] == rating_event_id):
                label = e.payload["new_raw_label"]
        return label

    # -------------------------------------------------------------- 便捷写入
    def import_batch(self, rows: list[dict[str, Any]], categories: list[str],
                     *, batch_id: str, source: str = "",
                     note: str = "") -> list[Event]:
        """导入一批评价。rows 每项含 project/rater/period/raw_label。

        原始文件内容由调用方另存; 这里只登记事件。等级原文本逐字保留。
        """
        if not rows:
            raise StoreError("空批次: 没有可导入的评价")
        events: list[Event] = []
        events.append(Event(EventType.IMPORT_BATCH, {
            "type": EventType.IMPORT_BATCH.value,
            "ts": utc_now(),
            "batch_id": batch_id,
            "source": source,
            "categories": list(categories),
            "note": note,
        }))
        for i, row in enumerate(rows):
            for key in ("project", "rater", "period", "raw_label"):
                if key not in row:
                    raise StoreError(f"第 {i} 行缺少字段 {key}")
            events.append(Event(EventType.RATING, {
                "type": EventType.RATING.value,
                "ts": utc_now(),
                "event_id": f"{batch_id}-r{i:05d}",
                "batch_id": batch_id,
                "project": str(row["project"]),
                "rater": str(row["rater"]),
                "period": int(row["period"]),
                "raw_label": str(row["raw_label"]),
                "note": str(row.get("note", "")),
            }))
        # 先整体校验再落盘, 避免半批次
        for ev in events:
            self._validate_new(ev)
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as fh:
                for ev in events:
                    fh.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self._events.extend(events)
            self._rebuild_versions_file()
        return events

    def correct_rating(self, target_event_id: str, new_raw_label: str,
                       *, reason: str = "", operator: str = "") -> Event:
        """追加一条误点更正 (永不修改或删除旧事件)。"""
        ev = Event(EventType.CORRECTION, {
            "type": EventType.CORRECTION.value,
            "ts": utc_now(),
            "event_id": f"cor-{len(self._events):06d}-{target_event_id}",
            "target_event_id": target_event_id,
            "new_raw_label": str(new_raw_label),
            "reason": reason,
            "operator": operator,
        })
        return self.append(ev)

    # ------------------------------------------------------------------ 投影
    def categories(self) -> Optional[list[str]]:
        for e in self._events:
            if e.type is EventType.IMPORT_BATCH:
                return list(e.payload["categories"])
        return None

    def project(self, *, n_events: Optional[int] = None,
                version_id: Optional[str] = None
                ) -> "Projection":
        """把事件日志投影为当前有效评价集合。

        n_events / version_id 用于复现历史版本; 默认投影到最新。
        """
        events = self._events
        if n_events is not None:
            events = events[:n_events]
        categories: Optional[list[str]] = None
        labels_by_rating: dict[str, str] = {}
        correction_info: dict[str, tuple[str, str]] = {}
        records: list[RatingRecord] = []
        rating_meta: dict[str, dict[str, Any]] = {}
        for e in events:
            if e.type is EventType.IMPORT_BATCH:
                categories = list(e.payload["categories"])
            elif e.type is EventType.RATING:
                p = e.payload
                labels_by_rating[p["event_id"]] = p["raw_label"]
                rating_meta[p["event_id"]] = p
            elif e.type is EventType.CORRECTION:
                p = e.payload
                labels_by_rating[p["target_event_id"]] = p["new_raw_label"]
                correction_info[p["target_event_id"]] = (
                    p["event_id"], p.get("reason", ""))
        if categories is None:
            return Projection(categories=[], records=[], version_id=None,
                              n_events=len(events), n_corrections=0)
        for eid, p in rating_meta.items():
            if eid not in labels_by_rating:
                continue  # 理论上不会发生
            effective = labels_by_rating[eid]
            cor = correction_info.get(eid)
            records.append(RatingRecord(
                event_id=eid,
                project=p["project"],
                rater=p["rater"],
                period=p["period"],
                raw_label=p["raw_label"],
                effective_label=effective,
                corrected=cor is not None,
                correction_event_id=cor[0] if cor else None,
                correction_reason=cor[1] if cor else None,
                ts=p["ts"],
            ))
        version_id_real = hash_events(events)
        if version_id is not None and version_id != version_id_real:
            raise StoreError(
                f"版本不匹配: 请求 {version_id}, 实际投影 {version_id_real}")
        n_corr = sum(1 for e in events if e.type is EventType.CORRECTION)
        return Projection(categories=categories, records=records,
                          version_id=version_id_real,
                          n_events=len(events), n_corrections=n_corr)

    # ------------------------------------------------------------------ 版本
    def _rebuild_versions_file(self) -> None:
        """版本元数据可由事件日志完全派生, 启动时重建索引文件。

        版本边界与写入操作一致: 每个 IMPORT_BATCH 与其后的 RATING 属于
        同一批次版本; 每条 CORRECTION 各自产生一个版本。
        """
        lines = []
        prev_hash: Optional[str] = None
        boundaries = self._version_boundaries()
        for k in boundaries:
            sub = self._events[:k]
            cur_hash = hash_events(sub)
            n_corr = sum(1 for e in sub if e.type is EventType.CORRECTION)
            n_rat = sum(1 for e in sub if e.type is EventType.RATING)
            lines.append(json.dumps({
                "version_id": cur_hash,
                "parent_id": prev_hash,
                "n_events": k,
                "n_ratings": n_rat,
                "n_corrections": n_corr,
                "categories": self.categories() or [],
                "created_ts": "",
            }, ensure_ascii=False))
            prev_hash = cur_hash
        tmp = self.versions_path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""),
                       encoding="utf-8")
        tmp.replace(self.versions_path)

    def _version_boundaries(self) -> list[int]:
        """版本边界 (事件前缀长度)。"""
        boundaries: list[int] = []
        n = len(self._events)
        for i, e in enumerate(self._events):
            if e.type is EventType.IMPORT_BATCH:
                # 找到该批次最后一条 RATING
                j = i + 1
                while j < n and self._events[j].type is EventType.RATING:
                    j += 1
                boundaries.append(j)
            elif e.type is EventType.CORRECTION:
                boundaries.append(i + 1)
        return boundaries

    def versions(self) -> list[DataVersion]:
        out = []
        if not self.versions_path.exists():
            return out
        for line in self.versions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out.append(DataVersion(**d))
        return out

    def n_for_version(self, version_id: str) -> Optional[int]:
        for v in self.versions():
            if v.version_id == version_id:
                return v.n_events
        return None

    # ------------------------------------------------------------ 拟合/报告 IO
    def save_fit(self, fit_id: str, payload: dict[str, Any]) -> Path:
        path = self.fits_dir / f"{fit_id}.json"
        if path.exists():
            # 同版本同配置的重复拟合是幂等的; 不同内容则拒绝覆盖
            existing = json.loads(path.read_text(encoding="utf-8"))
            if (existing.get("version_id") == payload.get("version_id")
                    and existing.get("config") == payload.get("config")):
                return path
            raise StoreError(f"拟合产物已存在且内容不同 (不可变): {fit_id}")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(path)
        return path

    def load_fit(self, fit_id: str) -> dict[str, Any]:
        return json.loads((self.fits_dir / f"{fit_id}.json").read_text(
            encoding="utf-8"))

    def list_fits(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.fits_dir.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({"fit_id": d.get("fit_id", p.stem),
                        "version_id": d.get("version_id"),
                        "kind": d.get("config", {}).get("kind"),
                        "created_ts": d.get("created_ts"),
                        "status": d.get("status")})
        return out

    def save_report(self, report_id: str, payload: dict[str, Any]) -> Path:
        path = self.reports_dir / f"{report_id}.json"
        if path.exists():
            raise StoreError(f"报告已存在 (不可变): {report_id}")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(path)
        return path

    def load_report(self, report_id: str) -> dict[str, Any]:
        return json.loads((self.reports_dir / f"{report_id}.json").read_text(
            encoding="utf-8"))

    def list_reports(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.reports_dir.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({"report_id": d.get("report_id", p.stem),
                        "version_id": d.get("version_id"),
                        "fit_id": d.get("fit_id"),
                        "created_ts": d.get("created_ts"),
                        "valid": d.get("valid")})
        return out

    def archive_import_file(self, batch_id: str, filename: str,
                            content: bytes) -> Path:
        """把不可信的原始导入文件隔离保存到批次专属目录。"""
        bdir = self.imports_dir / batch_id
        bdir.mkdir(parents=True, exist_ok=True)
        dest = bdir / Path(filename).name
        dest.write_bytes(content)
        return dest


class Projection:
    """事件日志的只读投影。"""

    def __init__(self, categories: list[str], records: list[RatingRecord],
                 version_id: Optional[str], n_events: int,
                 n_corrections: int):
        self.categories = categories
        self.records = records
        self.version_id = version_id
        self.n_events = n_events
        self.n_corrections = n_corrections

    # ---- 派生索引 ----
    @property
    def K(self) -> int:
        """等级数。"""
        return len(self.categories)

    def label_to_index(self) -> dict[str, int]:
        return {lab: i for i, lab in enumerate(self.categories)}

    def projects(self) -> list[str]:
        return sorted({r.project for r in self.records})

    def raters(self) -> list[str]:
        return sorted({r.rater for r in self.records})

    def periods(self) -> list[int]:
        return sorted({r.period for r in self.records})

    def index(self) -> "ModelIndex":
        return ModelIndex.build(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "n_events": self.n_events,
            "n_corrections": self.n_corrections,
            "categories": self.categories,
            "records": [r.to_dict() for r in self.records],
        }


class ModelIndex:
    """把文本标识映射为模型使用的整数下标。等级结果保留原文本。"""

    def __init__(self, projects: list[str], raters: list[str],
                 periods: list[int], categories: list[str]):
        self.projects = projects
        self.raters = raters
        self.periods = periods
        self.categories = categories
        self.project_idx = {p: i for i, p in enumerate(projects)}
        self.rater_idx = {r: i for i, r in enumerate(raters)}
        self.period_idx = {t: i for i, t in enumerate(periods)}
        self.label_idx = {c: i for i, c in enumerate(categories)}

    @classmethod
    def build(cls, proj: Projection) -> "ModelIndex":
        return cls(proj.projects(), proj.raters(), proj.periods(),
                   list(proj.categories))
