"""核心数据结构与事件定义。

设计要点
--------
* 一切输入都以 *事件* 形式追加到日志, 原始等级文本 (raw_label) 永不被覆盖。
* 误点更正 = 一条 correction 事件把旧 rating 标记失效 + 一条新 rating 事件;
  旧文本与旧事件始终保留, 报告按数据版本 (事件序列哈希) 复现。
* RatingRecord 是事件日志在某一版本上的 *投影* (只读快照), 不是事实来源。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    IMPORT_BATCH = "import_batch"   # 一次导入批次 (登记有序等级文本)
    RATING = "rating"               # 一条评价
    CORRECTION = "correction"       # 对某条评价的误点更正 (只追加)


# 事件允许字段与必填字段
_EVENT_SCHEMA: dict[str, tuple[set[str], set[str]]] = {
    EventType.IMPORT_BATCH: (
        {"type", "ts", "batch_id", "source", "categories", "note"},
        {"type", "ts", "batch_id", "categories"},
    ),
    EventType.RATING: (
        {"type", "ts", "event_id", "batch_id", "project", "rater",
         "period", "raw_label", "note"},
        {"type", "ts", "event_id", "project", "rater", "period", "raw_label"},
    ),
    EventType.CORRECTION: (
        {"type", "ts", "event_id", "target_event_id", "new_raw_label",
         "reason", "operator"},
        {"type", "ts", "event_id", "target_event_id", "new_raw_label"},
    ),
}


def canonical_json(obj: Any) -> str:
    """稳定序列化: 键排序、无多余空白, 用于版本哈希。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Event:
    """不可变事件。"""
    type: EventType
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        etype = d.get("type")
        try:
            et = EventType(etype)
        except ValueError as exc:
            raise ValueError(f"未知事件类型: {etype!r}") from exc
        allowed, required = _EVENT_SCHEMA[et]
        missing = required - d.keys()
        if missing:
            raise ValueError(f"事件 {etype} 缺少必填字段: {sorted(missing)}")
        unknown = d.keys() - allowed
        if unknown:
            raise ValueError(f"事件 {etype} 含未知字段: {sorted(unknown)}")
        if et is EventType.IMPORT_BATCH:
            cats = d["categories"]
            if not isinstance(cats, list) or len(cats) < 2:
                raise ValueError("categories 必须是长度 >= 2 的有序列表")
            if len(set(cats)) != len(cats):
                raise ValueError("categories 存在重复等级文本")
        if et is EventType.RATING and not isinstance(d["period"], int):
            raise ValueError("period 必须是整数期次")
        return cls(et, dict(d))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    @property
    def event_id(self) -> Optional[str]:
        return self.payload.get("event_id")


@dataclass(frozen=True)
class Project:
    project_id: str
    name: str = ""


@dataclass(frozen=True)
class Rater:
    rater_id: str
    name: str = ""


@dataclass(frozen=True)
class RatingRecord:
    """某数据版本下的一条 *有效* 评价 (投影结果, 只读)。

    raw_label 永远保留首次录入的原文本; effective_label 是更正后的文本,
    未更正时二者相同。corrected=True 表示该记录经过追加更正。
    """
    event_id: str
    project: str
    rater: str
    period: int
    raw_label: str
    effective_label: str
    corrected: bool
    correction_event_id: Optional[str] = None
    correction_reason: Optional[str] = None
    ts: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "project": self.project,
            "rater": self.rater,
            "period": self.period,
            "raw_label": self.raw_label,
            "effective_label": self.effective_label,
            "corrected": self.corrected,
            "correction_event_id": self.correction_event_id,
            "correction_reason": self.correction_reason,
            "ts": self.ts,
        }


@dataclass
class DataVersion:
    """事件日志在某个前缀上的版本。"""
    version_id: str
    parent_id: Optional[str]
    n_events: int
    n_ratings: int
    n_corrections: int
    categories: list[str]
    created_ts: str

    def to_dict(self) -> dict[str, Any]:
        return field_dict(self)


def field_dict(obj: Any) -> dict[str, Any]:
    return {k: getattr(obj, k) for k in obj.__dataclass_fields__}


def hash_events(events: list[Event]) -> str:
    """对事件序列计算内容哈希 (版本号)。"""
    h = hashlib.sha256()
    for ev in events:
        h.update(canonical_json(ev.to_dict()).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]
