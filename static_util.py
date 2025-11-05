import base64
import io

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

import color_style as color
from graph_util import (
    get_line_points,
    split_line_into_segments,
    find_wave_peaks,
    find_wave_troughs,
    find_downtrend_lines_from_peaks,
    find_uptrend_lines_from_troughs,
)


matplotlib.use("Agg")


def generate_kline_image(kline_data) -> dict:
    """
    Generate a candlestick (K-line) chart from OHLCV data, save it locally, and return a base64-encoded image.

    Args:
        kline_data (dict): Dictionary with keys including 'Datetime', 'Open', 'High', 'Low', 'Close'.
        filename (str): Name of the file to save the image locally (default: 'kline_chart.png').

    Returns:
        dict: Dictionary containing base64-encoded image string and local file path.
    """

    df = pd.DataFrame(kline_data)
    # Use all data passed in - no limit, let days parameter control the data range

    df.to_csv("record.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    try:
        # df.index = pd.to_datetime(df["Datetime"])
        df.index = pd.to_datetime(df["Datetime"], format="%Y-%m-%d %H:%M:%S")

    except ValueError:
        print("ValueError at graph_util.py\n")

    # Save image locally
    fig, axlist = mpf.plot(
        df[["Open", "High", "Low", "Close"]],
        type="candle",
        style=color.my_color_style,
        figsize=(12, 6),
        returnfig=True,
        block=False,
    )
    axlist[0].set_ylabel("Price", fontweight="normal")
    axlist[0].set_xlabel("Datetime", fontweight="normal")

    fig.savefig(
        fname="kline_chart.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)
    # ---------- Encode to base64 -----------------
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=600, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)  # release memory

    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "pattern_image": img_b64,
        "pattern_image_description": "Candlestick chart saved locally and returned as base64 string.",
    }


def generate_trend_image(kline_data) -> dict:
    """
    Generate a candlestick chart with trendlines from OHLCV data,
    save it locally as 'trend_graph.png', and return a base64-encoded image.

    Returns:
        dict: base64 image and description
    """
    data = pd.DataFrame(kline_data)
    # Use all data passed in - no limit, let days parameter control the data range
    candles = data.copy()

    candles["Datetime"] = pd.to_datetime(candles["Datetime"])
    candles.set_index("Datetime", inplace=True)

    all_segments = []
    colors = []
    apds = []
    
    # 补充方法：基于收盘价的完整波段高点识别下降趋势线
    # 使用波浪理论识别完整波段高点，而不是简单的局部高点
    try:
        close_series = candles["Close"].values
        high_series = candles["High"].values
        # 根据数据量自适应调整窗口大小
        # 对于日线数据，窗口约20；对于周线数据，窗口约4-5；对于月线数据，窗口约2-3
        data_len = len(close_series)
        # 基准窗口：假设日线数据约200-250个点，窗口为20
        # 按比例缩放窗口
        base_window = 20
        base_data_len = 200
        adaptive_window = max(5, int(base_window * data_len / base_data_len))
        adaptive_window = min(adaptive_window, data_len // 4)  # 最多不超过数据长度的1/4
        
        # 使用波浪理论识别完整波段高点
        # 使用最高价来识别波段高点的位置，然后取该位置的收盘价来画趋势线
        # min_amplitude_ratio=0.03 可以识别更多完整的波段高点，同时过滤掉噪音
        peaks = find_wave_peaks(high_series, close_data=close_series, min_amplitude_ratio=0.03, lookback_window=adaptive_window, lookforward_window=adaptive_window)
        downtrend_lines = find_downtrend_lines_from_peaks(peaks, min_points=3, min_slope=-0.001)
        
        if downtrend_lines:
            print(f"[DEBUG] 找到 {len(downtrend_lines)} 条下降趋势线（基于局部高点）")
            for idx, (slope, intercept, point_indices) in enumerate(downtrend_lines):
                # 生成趋势线的y值
                x_indices = np.arange(len(candles))
                trend_y = slope * x_indices + intercept
                
                # 转换为时间锚定的坐标点
                trend_seq = get_line_points(candles, trend_y)
                trend_segments = split_line_into_segments(trend_seq)
                all_segments.extend(trend_segments)
                colors.extend(["red"] * len(trend_segments))  # 使用红色表示下降趋势线
                
                # 添加到 addplot
                if idx == 0:
                    apds.append(mpf.make_addplot(
                        trend_y, color="red", width=2, linestyle="--", label="Downtrend (Wave Peaks)"
                    ))
                else:
                    apds.append(mpf.make_addplot(
                        trend_y, color="red", width=2, linestyle="--"
                    ))
                print(f"[DEBUG]   下降趋势线 {idx+1}: 连接 {len(point_indices)} 个点，斜率={slope:.4f}")
    except Exception as e:
        print(f"[DEBUG] 下降趋势线识别失败: {e}")
    
    # 补充方法：基于收盘价的完整波段低点识别上升趋势线
    # 使用波浪理论识别完整波段低点，连接三个或多个相邻的回调低点
    try:
        close_series = candles["Close"].values
        low_series = candles["Low"].values
        # 根据数据量自适应调整窗口大小（与下降趋势线使用相同的自适应逻辑）
        data_len = len(close_series)
        base_window = 20
        base_data_len = 200
        adaptive_window = max(5, int(base_window * data_len / base_data_len))
        adaptive_window = min(adaptive_window, data_len // 4)  # 最多不超过数据长度的1/4
        
        # 使用波浪理论识别完整波段低点
        # 使用最低价来识别波段低点的位置，然后取该位置的收盘价来画趋势线
        # min_amplitude_ratio=0.03 可以识别更多完整的波段低点，同时过滤掉噪音
        troughs = find_wave_troughs(low_series, close_data=close_series, min_amplitude_ratio=0.03, lookback_window=adaptive_window, lookforward_window=adaptive_window)
        uptrend_lines = find_uptrend_lines_from_troughs(troughs, min_points=3, min_slope=0.001)
        
        if uptrend_lines:
            print(f"[DEBUG] 找到 {len(uptrend_lines)} 条上升趋势线（基于完整波段低点）")
            for idx, (slope, intercept, point_indices) in enumerate(uptrend_lines):
                # 生成趋势线的y值
                x_indices = np.arange(len(candles))
                trend_y = slope * x_indices + intercept
                
                # 转换为时间锚定的坐标点
                trend_seq = get_line_points(candles, trend_y)
                trend_segments = split_line_into_segments(trend_seq)
                all_segments.extend(trend_segments)
                colors.extend(["green"] * len(trend_segments))  # 使用绿色表示上升趋势线
                
                # 添加到 addplot
                if idx == 0:
                    apds.append(mpf.make_addplot(
                        trend_y, color="green", width=2, linestyle="--", label="Uptrend (Wave Troughs)"
                    ))
                else:
                    apds.append(mpf.make_addplot(
                        trend_y, color="green", width=2, linestyle="--"
                    ))
                print(f"[DEBUG]   上升趋势线 {idx+1}: 连接 {len(point_indices)} 个点，斜率={slope:.4f}")
    except Exception as e:
        print(f"[DEBUG] 上升趋势线识别失败: {e}")
    
    # Generate figure with legend and save locally
    # 构建 mpf.plot 的参数，只有当列表不为空时才传递
    plot_kwargs = {
        "type": "candle",
        "style": color.my_color_style,
        "returnfig": True,
        "figsize": (12, 6),
        "block": False,
    }
    if apds:  # 只有当 apds 不为空时才传递
        plot_kwargs["addplot"] = apds
    if all_segments:  # 只有当 all_segments 不为空时才传递
        plot_kwargs["alines"] = dict(alines=all_segments, colors=colors, linewidths=1)
    
    fig, axlist = mpf.plot(candles, **plot_kwargs)

    axlist[0].set_ylabel("Price", fontweight="normal")
    axlist[0].set_xlabel("Datetime", fontweight="normal")

    # save fig locally
    fig.savefig(
        "trend_graph.png", format="png", dpi=600, bbox_inches="tight", pad_inches=0.1
    )
    plt.close(fig)

    # Add legend manually
    axlist[0].legend(loc="upper left")

    # Save to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return {
        "trend_image": img_b64,
        "trend_image_description": "Trend-enhanced candlestick chart with support/resistance lines.",
    }
