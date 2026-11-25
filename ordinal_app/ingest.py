"""CSV 导入。

安全约定 (工作区中的外部文件均为不可信数据):
* 只依赖标准库 csv, 不执行文件中的任何内容;
* 原始文件由 CLI 复制到 data/imports/<batch_id>/ 隔离保存, 解析逻辑与
  文件位置无关, 应用代码不从导入目录运行。
等级文本 (raw_label) 逐字保留, 不做任何数值化; 类别顺序在首次导入时登记。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

from .store import EventStore

REQUIRED_COLUMNS = {"project", "rater", "period", "raw_label"}


def parse_csv(path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """解析导入文件。

    支持两种形式:
    1. 普通表: project,rater,period,raw_label[,note]
    2. 首行以 '#' 开头声明等级体系, 例如:
       # categories,很差,较差,一般,满意,很满意
       未声明时由数据中首次出现的顺序推断, 并在返回值中给出。
    """
    path = Path(path)
    categories: Optional[list[str]] = None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        all_lines = fh.readlines()
    # 跳过所有 '#' 注释行; 其中 '# categories,...' 声明有序等级体系
    data_lines: list[str] = []
    for line in all_lines:
        if line.startswith("#"):
            parts = next(csv.reader([line.lstrip("#").strip()]))
            if parts and parts[0].strip() == "categories":
                categories = [c.strip() for c in parts[1:] if c.strip()]
            continue
        data_lines.append(line)
    reader = csv.DictReader(data_lines)
    if reader.fieldnames is None:
        raise ValueError("导入文件为空或缺少表头")
    cols = set(reader.fieldnames)
    missing = REQUIRED_COLUMNS - cols
    if missing:
        raise ValueError(f"缺少必填列: {sorted(missing)}")
    seen_order: list[str] = []
    for ln, row in enumerate(reader, 2):
        if not any((v or "").strip() for v in row.values()):
            continue
        try:
            period = int(str(row["period"]).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"第 {ln} 行 period 不是整数: "
                             f"{row.get('period')!r}") from exc
        label = str(row["raw_label"]).strip()
        if not label:
            raise ValueError(f"第 {ln} 行 raw_label 为空")
        if categories is None and label not in seen_order:
            seen_order.append(label)
        rows.append({
            "project": str(row["project"]).strip(),
            "rater": str(row["rater"]).strip(),
            "period": period,
            "raw_label": label,
            "note": str(row.get("note", "") or "").strip(),
        })
    if categories is None:
        categories = seen_order
    if len(categories) < 2:
        raise ValueError("至少需要 2 个不同等级才能建立有序尺度")
    unknown = sorted({r["raw_label"] for r in rows} - set(categories))
    if unknown:
        raise ValueError(f"存在未在等级体系中登记的等级文本: {unknown}")
    return rows, categories


def import_file(store: EventStore, csv_path: str | Path, *,
                batch_id: str, note: str = "") -> dict[str, Any]:
    """解析 -> 隔离归档原件 -> 追加事件。"""
    rows, categories = parse_csv(csv_path)
    store.archive_import_file(batch_id, Path(csv_path).name,
                              Path(csv_path).read_bytes())
    events = store.import_batch(rows, categories, batch_id=batch_id,
                                source=str(csv_path), note=note)
    return {"batch_id": batch_id, "n_rows": len(rows),
            "categories": categories,
            "n_events": len(events),
            "version_id": store.project().version_id}
