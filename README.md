# 有序等级数据分析应用 (Ordinal Rating Scale Analysis)

服务评价通常只有 **“很差 / 较差 / 一般 / 满意 / 很满意”** 这类**有序类别**，
不同评价者的打分尺度还可能不同。本应用把等级当作有序类别、联合估计
**项目表现（潜在能力）** 与 **评价者阈值**，而不是把等级当成等间距数字做平均。

## 它解决什么问题

1. **等级不是等间距数字** —— 用累积链接有序模型（cumulative link / graded
   response），`P(Y≤k)=F(τ_k − b)`，等级只通过排序进入模型。
2. **评价者尺度不同** —— 每个评价者有自己的一组阈值；阈值用
   “位置 + softplus 间隙”参数化，**从构造上严格有序、绝不交叉**。
3. **尺度分离（separation）** —— 两组项目若从不共享评价者，其相对位置没有
   似然信息。应用用“项目-项目重叠图”的连通分量**明确报告分离、不可跨组比较**，
   而不是给出看似精确实则由先验决定的数字。
4. **固定 vs 缓变阈值** —— 支持固定阈值与随时期随机游走的缓变阈值，用
   **整项目留出 + 滚动起源**（不用未来评价校正过去）选择。
5. **误点更正可追溯、旧报告可复现** —— 一切输入进 append-only 事件日志；
   更正只追加，原始等级文本逐字保留；报告绑定数据版本哈希，随时可重放复现。
6. **网页解释** —— 查看同一项目在不同评价者眼中的等级概率、调整锚定项目、
   对比阈值是否随时期漂移。

## 目录结构

```
ordinal_app/
  models.py       数据结构与不可变事件
  store.py        append-only 事件日志、版本投影、拟合/报告持久化
  ingest.py       CSV 导入 (原文本逐字保留, 原件隔离归档)
  stats.py        累积链接模型: 概率、对数似然、解析梯度、不交叉阈值
  fit.py          MAP 拟合 (多起点 L-BFGS-B)、Hessian 标准误、概率表
  diagnostics.py  重叠图分离、缺等级、尺度反转、阈值漂移
  synth.py        已知生成参数的虚构样例 (base/reversal/separation/missing/tiny)
  select.py       整项目留出与滚动起源模型选择
  report.py       不可变报告快照与复现核对
  web.py          只读 HTTP/JSON API (标准库)
  web_static/     原生 HTML/SVG 前端 (无框架、无第三方 JS)
  cli.py          命令行入口
tests/            unittest 测试 (28 个, 无第三方测试依赖)
examples/         CSV 导入样例
```

## 快速开始

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy scipy

# 1) 初始化空应用目录
python -m ordinal_app.cli init --root ./work

# 2) 导入虚构样例 (也可用 ingest 导入真实 CSV)
python -m ordinal_app.cli synth --root ./work --scenario base --batch b1
#   场景: base(正常漂移) / reversal(尺度反转) / separation(完全分离)
#         missing(缺等级) / tiny(手工似然夹具)

# 3) 拟合固定阈值与缓变阈值
python -m ordinal_app.cli fit --root ./work --kind fixed
python -m ordinal_app.cli fit --root ./work --kind drift

# 4) 诊断 (分离/缺等级/反转/不稳定阈值)
python -m ordinal_app.cli diagnose --root ./work

# 5) 模型选择 (整项目留出 + 滚动起源, 无未来泄漏)
python -m ordinal_app.cli select --root ./work

# 6) 生成不可变报告
python -m ordinal_app.cli report --root ./work --kind fixed

# 7) 追加一次误点更正 (旧事件与原文本都保留)
python -m ordinal_app.cli correct --root ./work \
    --event b1-r00005 --new-label 很满意 --reason "人工核对发现误点"

# 8) 按版本哈希复现旧报告
python -m ordinal_app.cli versions --root ./work
python -m ordinal_app.cli reproduce --root ./work --report <report_id>

# 9) 网页
python -m ordinal_app.cli serve --root ./work --port 8000
#   打开 http://127.0.0.1:8000/
```

导入真实 CSV（首行可用 `# categories,...` 声明有序等级体系，列：
`project,rater,period,raw_label[,note]`）：

```bash
python -m ordinal_app.cli ingest --root ./work \
    --file examples/ratings_sample.csv --batch csv1
```

## 统计模型

对项目 p 的潜在能力 `b_p`（零和锚识别）与评价者 r 在时期 t 的阈值
`τ_{r,t,0}<…<τ_{r,t,K−2}`：

```
P(Y ≤ k | p,r,t) = F(τ_{r,t,k} − b_p)
P(Y = k)         = F(τ_k − b_p) − F(τ_{k−1} − b_p)
```

- `F` 为 logistic（默认）或标准正态 CDF（`--link probit`）。
- 阈值参数化为 `位置 + cumsum(softplus(间隙))`，间隙恒正 ⇒ 阈值严格不交叉。
- 缓变模型的位置为随机游走 `a_{r,t}=a_{r,0}+Σ w`，`w~N(0, σ_walk)`；
  `σ_walk` 越大越接近自由阈值，越大惩罚越强越接近固定阈值。
- 全部自由参数带弱信息先验，**完全分离时后验仍正常**（MLE 会发散到无穷），
  此时标准误很大并由诊断标记。

## 验证与测试

```bash
python -m unittest discover -s tests -v
```

- `test_likelihood.py`：tiny 夹具上手工复算概率/似然、解析梯度对有限差分、
  阈值随机参数下永不交叉。
- `test_pipeline.py`：事件溯源、追加更正与旧报告复现、参数回收、
  分离/缺等级/反转诊断、CSV 解析。
- `test_selection.py`：整项目留出确实不含被留项目、滚动起源不借未来数据。
- `test_web.py`：真实 HTTP 服务器上的 API 合约、概率归一、路径穿越拦截。

## 安全约定

- 原始导入文件被复制到 `data/imports/<batch_id>/` 隔离保存，应用代码不从
  导入目录运行；读取外部数据建议用 `python -I`。
- 网页服务默认只监听 `127.0.0.1`，且为只读（没有任何写入/删除 API）。
- 事件与报告文件只追加、不可变；重复拟合同版本同配置幂等，内容不同则拒绝覆盖。
