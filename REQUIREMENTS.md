# 需求追溯说明

把原始需求逐条映射到实现与测试，便于核对“从空目录完成接收、拟合、诊断和解释”。

| # | 原始要求 | 实现位置 | 验证 |
|---|----------|----------|------|
| 1 | 等级只有“很差/一般/满意”等，评价者尺度可能不同；不能把等级当等间距数字平均 | `stats.py` 累积链接有序模型，等级仅作有序类别（`TensorData.y` 存下标，原文本另存）；每评价者独立阈值 | `tests/test_likelihood.py` 手工概率；`examples/ratings_sample.csv` |
| 2 | 联合估计项目表现与评价者阈值 | `fit.fit_model` 同时估能力 `b_p` 与阈值 `τ_{r,t,k}` | `test_base_parameter_recovery`（与已知生成参数相关≥0.85） |
| 3 | 有序阈值不能交叉 | `stats.ParamLayout.unpack`：`位置+cumsum(softplus(间隙))`，间隙恒正 | `test_thresholds_never_cross`（20 组随机自由参数） |
| 4 | 项目与评价者几乎不交叉时报告分离不开的尺度 | `diagnostics.overlap_components` 并查集 + `DiagnosticReport.separated`，明确文字说明不可跨分量比较 | `test_separation_detected`、`make_separation` |
| 5 | 支持固定阈值与随时期缓慢变化的阈值 | `FitConfig.kind=fixed/drift`，drift 用随机游走位置 + `sigma_walk` 惩罚 | 梯度测试 fixed/drift 两种；CLI `fit --kind` |
| 6 | 模型选择按完整项目留出，不借未来评价校正过去 | `select.leave_one_project_out`（整项目全部评价留出）、`select.rolling_origin`（≤t 训练预测 t+1） | `tests/test_selection.py` 两纪律断言 |
| 7 | 人工标记一次等级误点只追加更正 | `EventStore.correct_rating` 仅追加 `correction` 事件，无修改/删除接口 | `test_correction_is_append_only_and_reproducible` |
| 8 | 更正后相关拟合和稳定性分析失效，原报告仍能复现 | 报告绑定版本哈希与完整快照；`report.reproduce_report` 重放旧版本重拟合，`valid_now` 按当前日志实时判定 | 同上 + `test_web` 复现接口 |
| 9 | 等级结果保留原文本 | `RatingRecord.raw_label` 永不改，`effective_label` 为更正后值；数据页两列都展示 | 导入/更正测试与 `/api/data`、`data.html` |
| 10 | 网页查看同一项目在不同评价者眼中的等级概率 | `web.WebState.project_detail` + `project.html` 堆叠概率条（每段带 tooltip，总和=1） | `test_project_probabilities_sum_to_one` |
| 11 | 调整锚定项目 | 零和锚拟合；`project.html` 锚定选择器对能力与阈值做同步平移的重表达（不改变拟合） | `/api/anchor`、页面交互 |
| 12 | 对比阈值是否漂移 | `thresholds.html` 固定 vs 缓变阈值位置折线 + 相邻期 Δ/SE/z 显著性表；`diagnostics` drift_summary | `test_thresholds_and_selection` |
| 13 | 准备评价者尺度反转样例 | `synth.make_reversal`：反转者记录等级 k→K−1−k，已知生成参数存档 | `test_reversal_detected`（留一者共识相关≤−0.35） |
| 14 | 准备完全分离样例 | `synth.make_separation`：两组项目无共同评价者 | `test_separation_detected` |
| 15 | 准备缺等级样例 | `synth.make_missing_category`：相邻阈值贴近使中间等级确定性缺失（含断言） | `test_missing_category_detected` |
| 16 | 用已知生成参数核对 | 所有 `synth` 场景返回 `generating`（能力、阈值、链接、种子），CLI 落盘 `generating_params.json` | 参数回收测试 |
| 17 | 小数据似然核对 | `synth.make_tiny` 固定夹具 + `stats.log_likelihood_natural`，与手工 logit 公式逐项比对 | `TestTinyLikelihood`（含 probit 对正态 CDF） |
| 18 | 从空目录完成接收、拟合、诊断、解释 | `cli init/ingest|synth/fit/diagnose/select/report/correct/serve` 全流程 | README 快速开始；冒烟演练 |

## 关键设计决策与权衡

- **识别性锚定**：能力零和（`b_0=−Σb_j`）而非固定某项目为 0，这样任何项目
  都能在网页上被选为展示锚，且只是平移重表达。
- **MAP 而非 MLE**：弱信息先验让完全分离/缺等级时后验仍正常，避免 MLE 发散；
  代价是小样本收缩，故同时报告标准误与不稳定阈值清单。
- **选择主指标**：以滚动起源 MLPD（时间外推）为主、整项目留出为辅，因为它
  直接检验“不借未来”的预测质量；简约原则下 drift 不明显优于 fixed 即选 fixed。
- **反转检测**：用“留一者共识”的 Pearson 相关而非全局共识，避免反转者污染
  自身参照；并用重叠门控防止把尺度分离误判成反转。等级下标仅用于此诊断，不建模。
- **事件溯源**：版本元数据可由事件日志完全派生（启动重建），事实来源只有
  `events.jsonl`；报告/拟合文件是缓存式不可变产物。
