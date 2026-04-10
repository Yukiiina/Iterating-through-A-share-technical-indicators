# 32.py 内部代码逻辑树状结构

```text
32.py (全量因子防过拟合网格寻优脚本)
├── [Imports] 依赖库导入 
│   └── baostock, pandas, numpy, pandas_ta, matplotlib.pyplot, os, warnings
│
├── [Module 1] 模块 1：数据获取
│   └── def get_data(symbol, start_date)
│       ├── 识别输入格式 (自动补全 sh./sz. 前缀)
│       ├── 检查本地 CSV 缓存
│       ├── 调用 BaoStock API 拉取 OHLCV 数据
│       └── 清洗数据并存入本地缓存
│
├── [Module 2] 模块 2：全量指标与参数网格生成器
│   └── def generate_all_signals(df)
│       ├── 字典初始化: signals = {}
│       ├── A. 变周期参数网格循环 (N = 10, 20, 30, 40, 50, 60)
│       │   ├── 趋势类 (Trend_SMA, EMA, KAMA, DPO, AROON, ADX)
│       │   ├── 动量类 (Mom_MOM, BIAS, RSI, ROC, WR, CCI, CMO, TRIX)
│       │   ├── 波动类 (Vol_ATR_Brk, BBANDS, DC, ACCBANDS)
│       │   └── 成交量类 (Vol_MFI, EOM, FI)
│       └── B. 固定参数复杂指标计算
│           └── MACD_Std, SAR_Std, KDJ_Std, UO_Std, POS_Std, MASSI_Std, RVI_Std, AD_Std, OBV_Std
│
├── [Module 3] 模块 3：底层收益明细计算器
│   └── def calculate_daily_returns(df, signal_series, fee=0.0003)
│       ├── 仓位状态推导 (Shift 1 次日建仓)
│       ├── 交易行为记录 (用于扣除手续费)
│       └── 计算策略每日净收益 (Strategy_Return)
│
└── [Main] 主程序引擎
    └── if __name__ == "__main__":
        ├── 1. 设定测试标的与时间 (target_stock, start_date)
        ├── 2. 获取数据 (get_data)
        ├── 3. 切分样本内外时间线 (70% 训练集, 30% 盲测集)
        ├── 4. 批量生成策略信号 (generate_all_signals)
        ├── 5. 核心测评引擎 (遍历 all_signals)
        │   ├── 计算 IS_Alpha (样本内超额)
        │   ├── 计算 OOS_Alpha (样本外超额)
        │   └── 计算 Degradation (衰减度)
        ├── 6. 榜单输出器
        │   ├── 剔除负收益无效参数
        │   ├── 按衰减度从小到大排序 (robust_df)
        │   └── 打印终端控制台报表
        └── 7. 数据可视化模块 (matplotlib)
            ├── 绘制底层散点图 (全量灰色点阵)
            ├── 绘制上层散点图 (稳健红色点阵)
            ├── 绘制基准轴与对角线 (y=x 零衰减线)
            └── 渲染最终图表 (plt.show)
