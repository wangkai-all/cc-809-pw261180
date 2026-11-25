"""命令行入口: 从空目录完成接收、拟合、诊断与解释。

示例
----
python -m ordinal_app.cli init --root ./work
python -m ordinal_app.cli ingest --root ./work --file data.csv --batch b1
python -m ordinal_app.cli synth --root ./work --scenario base --batch s1
python -m ordinal_app.cli fit --root ./work --kind drift
python -m ordinal_app.cli diagnose --root ./work
python -m ordinal_app.cli select --root ./work
python -m ordinal_app.cli report --root ./work --kind fixed
python -m ordinal_app.cli correct --root ./work --event s1-r00003 \\
    --new-label 满意 --reason "误点更正"
python -m ordinal_app.cli versions --root ./work
python -m ordinal_app.cli reproduce --root ./work --report report-xxxx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import diagnostics, report, synth
from .fit import fit_model
from .ingest import import_file
from .select import compare_models
from .stats import FitConfig
from .store import EventStore


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _store(args) -> EventStore:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    return EventStore(root)


def cmd_init(args) -> None:
    store = _store(args)
    _print({"ok": True, "root": str(Path(args.root).resolve()),
            "events_file": str(store.events_path),
            "n_events": len(store.events)})


def cmd_ingest(args) -> None:
    store = _store(args)
    _print(import_file(store, args.file, batch_id=args.batch, note=args.note))


def cmd_synth(args) -> None:
    store = _store(args)
    builder = synth.SCENARIOS[args.scenario]
    if args.scenario == "tiny":
        ds = builder(link=args.link)
        # tiny 是概率单元夹具, 取每个 (项目,评价者) 的最高概率等级作为观测
        best: dict[tuple[str, str], dict] = {}
        for row in ds.rows:
            key = (row["project"], row["rater"])
            if key not in best or row["true_probability"] > \
                    best[key]["true_probability"]:
                best[key] = row
        rows = [{"project": r["project"], "rater": r["rater"],
                 "period": r["period"], "raw_label": r["raw_label"],
                 "note": "tiny 夹具: 取模态等级"}
                for r in best.values()]
    else:
        ds = builder(seed=args.seed, link=args.link)
        rows = [{"project": r["project"], "rater": r["rater"],
                 "period": r["period"], "raw_label": r["raw_label"],
                 "note": f"虚构数据({args.scenario}) 真实等级下标="
                         f"{r['true_category_index']}"}
                for r in ds.rows]
    store.import_batch(rows, ds.categories, batch_id=args.batch,
                       source=f"synthetic:{args.scenario}",
                       note=json.dumps(ds.generating, ensure_ascii=False))
    out_path = Path(args.root) / "data" / "imports" / args.batch / \
        "generating_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ds.to_dict(), ensure_ascii=False,
                                   indent=2), encoding="utf-8")
    _print({"batch_id": args.batch, "scenario": args.scenario,
            "n_rows": len(rows), "categories": ds.categories,
            "generating_params_saved": str(out_path),
            "version_id": store.project().version_id})


def cmd_fit(args) -> None:
    store = _store(args)
    proj = store.project()
    cfg = FitConfig(kind=args.kind, link=args.link,
                    sigma_walk=args.sigma_walk)
    result = fit_model(proj, cfg)
    path = store.save_fit(result.fit_id, result.to_dict())
    _print({"fit_id": result.fit_id, "saved": str(path),
            "converged": result.converged,
            "log_likelihood": result.log_likelihood,
            "warnings": result.warnings,
            "abilities": result.ability_table(),
            "n_threshold_rows": len(result.threshold_table())})


def cmd_diagnose(args) -> None:
    store = _store(args)
    proj = store.project()
    fit = None
    if args.fit_id:
        saved = store.load_fit(args.fit_id)
        fit = fit_model(proj, FitConfig(**saved["config"]))
    _print(diagnostics.diagnose(proj, fit).to_dict())


def cmd_select(args) -> None:
    store = _store(args)
    _print(compare_models(store.project(), link=args.link,
                          sigma_walk=args.sigma_walk))


def cmd_report(args) -> None:
    store = _store(args)
    payload = report.build_report(store, kind=args.kind, link=args.link,
                                  sigma_walk=args.sigma_walk,
                                  include_selection=not args.no_selection)
    _print({"report_id": payload["report_id"],
            "version_id": payload["version_id"],
            "valid": payload["valid"], "validity": payload["validity"],
            "diagnostic_problems":
                payload["diagnostics"]["problems"]})


def cmd_correct(args) -> None:
    store = _store(args)
    before = store.project()
    ev = store.correct_rating(args.event, args.new_label,
                              reason=args.reason, operator=args.operator)
    after = store.project()
    rec = next(r for r in after.records if r.event_id == args.event)
    _print({"correction_event": ev.to_dict(),
            "version_before": before.version_id,
            "version_after": after.version_id,
            "raw_label_kept": rec.raw_label,
            "effective_label": rec.effective_label,
            "corrected": rec.corrected,
            "note": "原文本与旧事件均保留; 既有报告按旧版本哈希仍可复现"})


def cmd_versions(args) -> None:
    store = _store(args)
    _print({"versions": [v.to_dict() for v in store.versions()],
            "fits": store.list_fits(), "reports": store.list_reports()})


def cmd_reproduce(args) -> None:
    store = _store(args)
    _print(report.reproduce_report(store, args.report))


def cmd_probs(args) -> None:
    store = _store(args)
    proj = store.project()
    cfg = FitConfig(kind=args.kind, link=args.link)
    result = fit_model(proj, cfg)
    rows = [r for r in result.probability_table(args.link)
            if r["project"] == args.project]
    _print({"project": args.project, "categories": proj.categories,
            "probabilities": rows})


def cmd_serve(args) -> None:
    from .web import serve
    store = _store(args)
    httpd = serve(store, host=args.host, port=args.port, link=args.link,
                  sigma_walk=args.sigma_walk)
    print(f"网页服务已启动: http://{args.host}:{args.port}/  (Ctrl-C 停止)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ordinal_app",
        description="有序等级 (评价) 数据分析: 累积链接模型 + 评价者阈值")
    sub = p.add_subparsers(dest="command", required=True)

    def add_root(sp):
        sp.add_argument("--root", required=True, help="应用数据目录")

    sp = sub.add_parser("init", help="初始化空应用目录")
    add_root(sp); sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("ingest", help="导入 CSV 评价文件")
    add_root(sp); sp.add_argument("--file", required=True)
    sp.add_argument("--batch", required=True); sp.add_argument("--note", default="")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("synth", help="导入虚构样例数据")
    add_root(sp); sp.add_argument("--scenario", required=True,
                                  choices=list(synth.SCENARIOS))
    sp.add_argument("--batch", required=True)
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--link", default="logit", choices=["logit", "probit"])
    sp.set_defaults(func=cmd_synth)

    sp = sub.add_parser("fit", help="拟合项目能力与评价者阈值")
    add_root(sp); sp.add_argument("--kind", default="fixed",
                                  choices=["fixed", "drift"])
    sp.add_argument("--link", default="logit", choices=["logit", "probit"])
    sp.add_argument("--sigma-walk", dest="sigma_walk", type=float, default=0.25)
    sp.set_defaults(func=cmd_fit)

    sp = sub.add_parser("diagnose", help="分离/缺等级/反转诊断")
    add_root(sp); sp.add_argument("--fit-id", default=None)
    sp.set_defaults(func=cmd_diagnose)

    sp = sub.add_parser("select", help="固定 vs 缓变阈值模型选择")
    add_root(sp); sp.add_argument("--link", default="logit",
                                  choices=["logit", "probit"])
    sp.add_argument("--sigma-walk", dest="sigma_walk", type=float, default=0.25)
    sp.set_defaults(func=cmd_select)

    sp = sub.add_parser("report", help="生成不可变报告快照")
    add_root(sp); sp.add_argument("--kind", default="fixed",
                                  choices=["fixed", "drift"])
    sp.add_argument("--link", default="logit", choices=["logit", "probit"])
    sp.add_argument("--sigma-walk", dest="sigma_walk", type=float, default=0.25)
    sp.add_argument("--no-selection", action="store_true")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("correct", help="追加误点更正 (不修改旧事件)")
    add_root(sp); sp.add_argument("--event", required=True)
    sp.add_argument("--new-label", required=True)
    sp.add_argument("--reason", default=""); sp.add_argument("--operator", default="")
    sp.set_defaults(func=cmd_correct)

    sp = sub.add_parser("versions", help="查看版本/拟合/报告清单")
    add_root(sp); sp.set_defaults(func=cmd_versions)

    sp = sub.add_parser("reproduce", help="按版本哈希复现旧报告")
    add_root(sp); sp.add_argument("--report", required=True)
    sp.set_defaults(func=cmd_reproduce)

    sp = sub.add_parser("probs", help="查看某项目在各评价者处的等级概率")
    add_root(sp); sp.add_argument("--project", required=True)
    sp.add_argument("--kind", default="fixed", choices=["fixed", "drift"])
    sp.add_argument("--link", default="logit", choices=["logit", "probit"])
    sp.set_defaults(func=cmd_probs)

    sp = sub.add_parser("serve", help="启动只读网页与 JSON API")
    add_root(sp); sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--link", default="logit", choices=["logit", "probit"])
    sp.add_argument("--sigma-walk", dest="sigma_walk", type=float, default=0.25)
    sp.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:  # noqa: BLE001
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
