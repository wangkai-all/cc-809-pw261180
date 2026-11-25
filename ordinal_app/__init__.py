"""有序等级数据分析应用 (ordinal rating scale analysis).

模块划分:
  models   数据结构与事件溯源
  store    append-only 事件日志 / 版本投影 / 拟合产物持久化
  stats    累积链接有序模型的概率与对数似然
  fit      MAP 拟合 (阈值严格不交叉), 固定 / 缓变阈值
  diagnostics  分离、缺等级、尺度反转等诊断
  synth    已知生成参数的虚构数据
  select   整项目留出与滚动起源模型选择
  web      只读网页与 JSON API
  cli      命令行入口
"""

__version__ = "1.0.0"
