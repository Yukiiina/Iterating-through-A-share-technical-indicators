# 量化回测与因子挖掘项目结构

```text
mom_quant/
├── 32.py                                     # 核心引擎：全量技术指标参数网格寻优与防过拟合盲测脚本
├── data_sh_000001_20160101_ohlcv_latest.csv  # 数据缓存：BaoStock 下载的上证指数全维度量价历史数据
├── requirements.txt                          # 依赖配置：项目所需的 Python 库 (baostock, pandas, pandas_ta, matplotlib 等)
└── STRUCTURE.md                              # 文档：项目目录结构说明 (本文件)
