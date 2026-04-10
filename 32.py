import baostock as bs
import pandas as pd
import numpy as np
import pandas_ta as ta
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 模块 1：数据获取
# ==========================================
def get_data(symbol, start_date):
    if symbol.startswith(('sh.', 'sz.')):
        bs_code = symbol
        clean_symbol = symbol.replace('.', '_')
    else:
        prefix = "sh" if symbol.startswith(('6', '9')) else "sz"
        bs_code = f"{prefix}.{symbol}"
        clean_symbol = symbol

    file_name = f"data_{clean_symbol}_{start_date}_ohlcv_latest.csv"
    
    if os.path.exists(file_name):
        print(f"正在从本地加载数据: {file_name}")
        return pd.read_csv(file_name, index_col='date', parse_dates=True)
    
    print(f"本地无缓存，正在连接 BaoStock 拉取 {bs_code} 数据...")
    bs.login()
    rs = bs.query_history_k_data_plus(
        bs_code, "date,open,high,low,close,volume", 
        start_date=f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}", 
        frequency="d", adjustflag="2"
    )
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()
    
    if not data_list: return pd.DataFrame()
        
    df = pd.DataFrame(data_list, columns=rs.fields)
    df['date'] = pd.to_datetime(df['date'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df = df.set_index('date').rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    df.to_csv(file_name)
    return df

# ==========================================
# 模块 2：全量指标与参数网格生成器
# ==========================================
def generate_all_signals(df):
    signals = {}
    close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']
    
    # ---------------------------------------------------------
    # A. 带有参数网格循环的指标 (N = 10, 20, 30, 40, 50, 60)
    # ---------------------------------------------------------
    for n in range(10, 61, 10):
        # --- 趋势类 ---
        try: signals[f'Trend_SMA_{n}'] = pd.Series(np.where(ta.sma(close, n) > ta.sma(close, n*3), 1, 0), index=df.index)
        except: pass
        try: signals[f'Trend_EMA_{n}'] = pd.Series(np.where(ta.ema(close, n) > ta.ema(close, n*3), 1, 0), index=df.index)
        except: pass
        try: signals[f'Trend_KAMA_{n}'] = pd.Series(np.where(ta.kama(close, n) > ta.kama(close, n*3), 1, 0), index=df.index)
        except: pass
        try: signals[f'Trend_DPO_{n}'] = pd.Series(np.where(ta.dpo(close, length=n) > 0, 1, 0), index=df.index)
        except: pass
        try: 
            aroon = ta.aroon(high, low, length=n)
            signals[f'Trend_AROON_{n}'] = pd.Series(np.where(aroon.iloc[:, 0] > aroon.iloc[:, 1], 1, 0), index=df.index)
        except: pass
        try: 
            adx = ta.adx(high, low, close, length=n)
            signals[f'Trend_ADX_{n}'] = pd.Series(np.where((adx.iloc[:, 0] > 25) & (adx.iloc[:, 1] > adx.iloc[:, 2]), 1, 0), index=df.index)
        except: pass

        # --- 动量类 ---
        try: signals[f'Mom_MOM_{n}'] = pd.Series(np.where(ta.mom(close, length=n) > 0, 1, 0), index=df.index)
        except: pass
        try: 
            sma_n = ta.sma(close, n)
            signals[f'Mom_BIAS_{n}'] = pd.Series(np.where(((close - sma_n) / sma_n) > 0, 1, 0), index=df.index)
        except: pass
        try: 
            rsi = ta.rsi(close, length=n)
            signals[f'Mom_RSI_{n}'] = pd.Series(np.where(rsi < 30, 1, np.where(rsi > 70, 0, np.nan)), index=df.index).ffill()
        except: pass
        try: signals[f'Mom_ROC_{n}'] = pd.Series(np.where(ta.roc(close, length=n) > 0, 1, 0), index=df.index)
        except: pass
        try: 
            wr = ta.willr(high, low, close, length=n)
            signals[f'Mom_WR_{n}'] = pd.Series(np.where(wr < -80, 1, np.where(wr > -20, 0, np.nan)), index=df.index).ffill()
        except: pass
        try: 
            cci = ta.cci(high, low, close, length=n)
            signals[f'Mom_CCI_{n}'] = pd.Series(np.where(cci < -100, 1, np.where(cci > 100, 0, np.nan)), index=df.index).ffill()
        except: pass
        try: signals[f'Mom_CMO_{n}'] = pd.Series(np.where(ta.cmo(close, length=n) > 0, 1, 0), index=df.index)
        except: pass
        try: signals[f'Mom_TRIX_{n}'] = pd.Series(np.where(ta.trix(close, length=n).iloc[:, 0] > 0, 1, 0), index=df.index)
        except: pass

        # --- 波动类 ---
        try: 
            atr = ta.atr(high, low, close, length=n)
            sma_base = ta.sma(close, 20) 
            signals[f'Vol_ATR_Brk_{n}'] = pd.Series(np.where(close > sma_base + atr, 1, np.where(close < sma_base, 0, np.nan)), index=df.index).ffill()
        except: pass
        try: 
            bb = ta.bbands(close, length=n)
            signals[f'Vol_BBANDS_{n}'] = pd.Series(np.where(close < bb.iloc[:, 0], 1, np.where(close > bb.iloc[:, 2], 0, np.nan)), index=df.index).ffill()
        except: pass
        try: 
            dc = ta.donchian(high, low, lower_length=n, upper_length=n)
            signals[f'Vol_DC_{n}'] = pd.Series(np.where(close >= dc.iloc[:, 2], 1, np.where(close <= dc.iloc[:, 0], 0, np.nan)), index=df.index).ffill()
        except: pass
        try: 
            acc = ta.accbands(high, low, close, length=n)
            signals[f'Vol_ACCBANDS_{n}'] = pd.Series(np.where(close < acc.iloc[:, 0], 1, np.where(close > acc.iloc[:, 2], 0, np.nan)), index=df.index).ffill()
        except: pass

        # --- 成交量类 ---
        try: 
            mfi = ta.mfi(high, low, close, vol, length=n)
            signals[f'Vol_MFI_{n}'] = pd.Series(np.where(mfi < 20, 1, np.where(mfi > 80, 0, np.nan)), index=df.index).ffill()
        except: pass
        try: signals[f'Vol_EOM_{n}'] = pd.Series(np.where(ta.eom(high, low, close, vol, length=n) > 0, 1, 0), index=df.index)
        except: pass
        try: signals[f'Vol_FI_{n}'] = pd.Series(np.where(ta.efi(close, vol, length=n) > 0, 1, 0), index=df.index)
        except: pass

    # ---------------------------------------------------------
    # B. 固定参数的经典复杂指标
    # ---------------------------------------------------------
    try:
        macd = ta.macd(close)
        macd_col = [c for c in macd.columns if c.startswith('MACDh')][0]
        signals['Trend_MACD_Std'] = pd.Series(np.where(macd[macd_col] > 0, 1, 0), index=df.index)
    except: pass
    try:
        sar = ta.psar(high, low, close)
        sar_col = [c for c in sar.columns if c.startswith('PSARl')][0]
        signals['Trend_SAR_Std'] = pd.Series(np.where(sar[sar_col].notna(), 1, 0), index=df.index)
    except: pass
    try:
        kdj = ta.kdj(high, low, close)
        signals['Mom_KDJ_Std'] = pd.Series(np.where(kdj.iloc[:, 0] > kdj.iloc[:, 1], 1, 0), index=df.index)
    except: pass
    try: signals['Mom_UO_Std'] = pd.Series(np.where(ta.uo(high, low, close) > 50, 1, 0), index=df.index)
    except: pass
    try: signals['Mom_POS_Std'] = pd.Series(np.where(ta.apo(close) > 0, 1, 0), index=df.index)
    except: pass
    try: 
        massi = ta.massi(high, low)
        signals['Vol_MASSI_Std'] = pd.Series(np.where((massi > 25) & (massi < massi.shift(1)), 1, 0), index=df.index)
    except: pass
    try: signals['Vol_RVI_Std'] = pd.Series(np.where(ta.rvi(close) > 50, 1, 0), index=df.index)
    except: pass
    try: 
        ad = ta.ad(high, low, close, vol)
        signals['Volume_AD_Std'] = pd.Series(np.where((ad > ad.shift(1)) & (close > ta.sma(close, 20)), 1, 0), index=df.index)
    except: pass
    try: 
        obv = ta.obv(close, vol)
        signals['Volume_OBV_Std'] = pd.Series(np.where((obv > obv.shift(1)) & (close > ta.sma(close, 20)), 1, 0), index=df.index)
    except: pass

    return signals

# ==========================================
# 模块 3：底层收益明细计算器
# ==========================================
def calculate_daily_returns(df, signal_series, fee=0.0003):
    data = df[['Close']].copy()
    data['Signal'] = signal_series.fillna(0)
    data['Position'] = data['Signal'].shift(1).fillna(0) 
    data['Trade_Action'] = data['Position'].diff().abs()
    
    data['Market_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Position'] * data['Market_Return'] - (data['Trade_Action'] * fee).fillna(0)
    return data

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    target_stock = "sh.000001"  
    start_date = "20180101" 
    
    df = get_data(target_stock, start_date)
    
    if not df.empty:
        split_idx = int(len(df) * 0.7)
        split_date = df.index[split_idx]
        
        print(f"\n[数据切分] 样本内训练: 至 {split_date.date()} | 样本外盲测: {split_date.date()} 至今")
        
        all_signals = generate_all_signals(df)
        results = []
        
        print(f"指标生成完毕！共计 {len(all_signals)} 组策略组合，正在进行防过拟合检验...")
        
        for name, sig in all_signals.items():
            daily_data = calculate_daily_returns(df, sig)
            
            is_data = daily_data.iloc[:split_idx]
            is_alpha = ((1 + is_data['Strategy_Return']).cumprod().iloc[-1] - 1)*100 - ((1 + is_data['Market_Return']).cumprod().iloc[-1] - 1)*100
            
            oos_data = daily_data.iloc[split_idx:]
            oos_alpha = ((1 + oos_data['Strategy_Return']).cumprod().iloc[-1] - 1)*100 - ((1 + oos_data['Market_Return']).cumprod().iloc[-1] - 1)*100
            
            degradation = is_alpha - oos_alpha
            
            results.append({
                'Indicator_Param': name, 
                'IS_Alpha(%)': round(is_alpha, 2),
                'OOS_Alpha(%)': round(oos_alpha, 2),
                'Degradation': round(degradation, 2)
            })
            
        res_df = pd.DataFrame(results)
        
        # 稳健指标筛选条件：样本外和样本内都必须大于0，且优先展示衰减度低的
        robust_df = res_df[(res_df['OOS_Alpha(%)'] > 0) & (res_df['IS_Alpha(%)'] > 0)].copy()
        robust_df = robust_df.sort_values(by='Degradation', ascending=True)
        
        print("\n" + "="*85)
        print(f" 全量稳健指标排行榜 (按过拟合衰减度 Degradation 从低到高排序)")
        print("="*85)
        if robust_df.empty:
            print("遗憾：未能找到在盲测期依然盈利的指标。系统性熊市摧毁了所有技术因子。")
        else:
            print(robust_df.head(20).to_string(index=False))
        print("="*85 + "\n")
        
        # -----------------------------------------------------
        # 可视化：全量指标样本内外散点图
        # -----------------------------------------------------
        fig, ax = plt.subplots(figsize=(14, 10)) # 稍微放大了画布以容纳更多文字
        
        # 绘制散点
        ax.scatter(res_df['IS_Alpha(%)'], res_df['OOS_Alpha(%)'], color='gray', alpha=0.4, label='Overfitted / Failed')
        
        if not robust_df.empty:
            ax.scatter(robust_df['IS_Alpha(%)'], robust_df['OOS_Alpha(%)'], color='crimson', edgecolors='black', s=60, label='Robust Strategies')
            
        # 遍历全量数据，为每一个点添加名称标注
        for i in range(len(res_df)):
            row = res_df.iloc[i]
            
            # 判断当前点是否属于“第一象限稳健点”
            is_robust = (row['OOS_Alpha(%)'] > 0) and (row['IS_Alpha(%)'] > 0)
            
            # 根据是否稳健，设置不同的文字样式，避免图表被文字彻底糊死
            text_color = 'darkblue' if is_robust else 'dimgray'
            font_size = 9 if is_robust else 6
            text_alpha = 1.0 if is_robust else 0.4
            
            ax.annotate(row['Indicator_Param'], 
                        (row['IS_Alpha(%)'], row['OOS_Alpha(%)']),
                        xytext=(4, 4), textcoords='offset points', 
                        fontsize=font_size, color=text_color, alpha=text_alpha)
        
        # 绘制基准线
        ax.axhline(0, color='black', lw=1.2, linestyle='--')
        ax.axvline(0, color='black', lw=1.2, linestyle='--')
        
        # 绘制 y=x 对角线
        min_val = min(res_df['IS_Alpha(%)'].min(), res_df['OOS_Alpha(%)'].min())
        max_val = max(res_df['IS_Alpha(%)'].max(), res_df['OOS_Alpha(%)'].max())
        ax.plot([min_val, max_val], [min_val, max_val], color='blue', alpha=0.3, label='Zero Degradation (y=x)')
        
        # 设置标题和图例
        ax.set_title('Grid Search: Out-of-Sample vs In-Sample Alpha (All Indicators Labeled)')
        ax.set_xlabel('In-Sample Alpha (%) - Training Period')
        ax.set_ylabel('Out-of-Sample Alpha (%) - Blind Test Period')
        
        # 填充背景色
        ax.axvspan(0, max_val, ymin=0.5, ymax=1, color='lightgreen', alpha=0.1) 
        ax.axvspan(0, max_val, ymin=0, ymax=0.5, color='lightcoral', alpha=0.1) 
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()