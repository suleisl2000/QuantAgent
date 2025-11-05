import base64
import io
from typing import Annotated

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import talib
from langchain_core.tools import tool

import color_style as color


matplotlib.use("Agg")


def find_wave_peaks(high_data, close_data=None, min_amplitude_ratio=0.05, lookback_window=20, lookforward_window=20):
    """
    根据波浪理论识别完整波段高点
    
    完整波段高点需要满足：
    1. 左侧有上升波段（从低点上升到此高点）
    2. 右侧有下降波段（从此高点下降到低点）
    3. 波幅足够大（高点与前后低点的差值占价格的比例达到最小要求）
    4. 在更大的范围内也是真正的峰值（不能被更高点"包住"）
    
    Args:
        high_data: 最高价序列，用于识别波段高点的位置
        close_data: 收盘价序列，用于获取高点的收盘价。如果为None，则使用high_data的值
        min_amplitude_ratio: 最小波幅比例（高点与前后低点的差值占高点价格的比例）
        lookback_window: 向前查找低点的窗口大小
        lookforward_window: 向后查找低点的窗口大小
    
    Returns:
        list of (index, value) tuples，每个元素是一个完整波段的高点
        - index: 高点的位置索引
        - value: 如果提供了close_data，则是该位置的收盘价；否则是high_data的值
    """
    if close_data is None:
        close_data = high_data
    
    peaks = []
    n = len(high_data)
    
    # 使用更大的窗口来检查是否被更高点"包住"
    larger_window = max(lookback_window, lookforward_window) * 2
    
    for i in range(lookback_window, n - lookforward_window):
        current_high = high_data[i]  # 使用最高价来判断是否为波段高点
        current_close = close_data[i]  # 使用收盘价作为返回的值
        
        # 查找左侧最低点（在lookback_window范围内，使用最低价来判断）
        # 注意：这里应该用low_data，但为了简化，我们先用close_data来近似
        # 如果需要更精确，应该传入low_data参数
        left_start = max(0, i - lookback_window)
        left_min_idx = np.argmin(close_data[left_start:i])  # 使用收盘价来查找低点
        left_min_idx = left_start + left_min_idx
        left_min_price = close_data[left_min_idx]
        
        # 查找右侧最低点（在lookforward_window范围内）
        right_end = min(n, i + lookforward_window + 1)
        right_min_idx = np.argmin(close_data[i+1:right_end])
        right_min_idx = i + 1 + right_min_idx
        right_min_price = close_data[right_min_idx]
        
        # 检查是否是一个完整的波段高点：
        # 1. 当前最高价必须高于左侧和右侧的低点
        # 2. 左侧必须有明显的上升（从左侧最低点到当前高点）
        # 3. 右侧必须有明显的下降（从当前高点右侧最低点）
        # 4. 波幅足够大
        
        left_rise = current_high - left_min_price
        right_fall = current_high - right_min_price
        
        # 计算波幅比例（相对于当前最高价）
        amplitude_ratio = min(left_rise, right_fall) / current_high if current_high > 0 else 0
        
        # 检查是否满足基本条件
        if (current_high > left_min_price and 
            current_high > right_min_price and
            left_rise > 0 and 
            right_fall > 0 and
            amplitude_ratio >= min_amplitude_ratio):
            
            # 额外检查：确保这是一个真正的峰值，而不是上升或下降过程中的点
            # 检查当前点是否在左侧和右侧都是局部最大值
            check_window = min(5, lookback_window // 4, lookforward_window // 4)
            is_local_max = True
            
            # 检查左侧是否有更高的点（使用最高价）
            if i > check_window:
                if max(high_data[i-check_window:i]) >= current_high:
                    is_local_max = False
            
            # 检查右侧是否有更高的点（使用最高价）
            if i < n - check_window:
                if max(high_data[i+1:i+check_window+1]) >= current_high:
                    is_local_max = False
            
            if not is_local_max:
                continue
            
            # 关键检查：在更大的范围内，确保不是被更高点"包住"的点
            # 检查左侧更大范围内是否有更高的点（使用最高价）
            larger_left_start = max(0, i - larger_window)
            larger_left_max = max(high_data[larger_left_start:i]) if i > larger_left_start else current_high
            larger_left_max_idx = larger_left_start + np.argmax(high_data[larger_left_start:i]) if i > larger_left_start else i
            
            # 检查右侧更大范围内是否有更高的点（使用最高价）
            larger_right_end = min(n, i + larger_window + 1)
            larger_right_max = max(high_data[i+1:larger_right_end]) if larger_right_end > i + 1 else current_high
            larger_right_max_idx = i + 1 + np.argmax(high_data[i+1:larger_right_end]) if larger_right_end > i + 1 else i
            
            # 如果左侧和右侧都有更高的点（且差距较大），说明这个点被"包住"了
            # 但只过滤明显被包住的情况（左右都有明显更高的点，且当前点不是趋势的一部分）
            left_higher = larger_left_max > current_high * 1.03  # 3%的容差
            right_higher = larger_right_max > current_high * 1.03
            
            # 如果右侧有更高的点且在较近范围内，说明当前点是上升过程中的点，不是完整波段高点
            if right_higher and larger_right_max_idx < i + lookforward_window:
                # 右侧有更高的点，且距离不远，说明当前点不是完整波段高点
                continue
            
            # 如果左侧有更高的点且在较近范围内，且右侧也有更高的点，说明当前点被包住了
            if left_higher and larger_left_max_idx > i - lookback_window and right_higher:
                # 左右都有更高的点，且都在较近范围内，说明当前点不是完整波段高点
                continue
            
            peaks.append((i, current_close))  # 返回收盘价，而不是最高价
    
    return peaks


def find_wave_troughs(low_data, close_data=None, min_amplitude_ratio=0.05, lookback_window=20, lookforward_window=20):
    """
    根据波浪理论识别完整波段低点
    
    完整波段低点需要满足：
    1. 左侧有下降波段（从高点下降到此低点）
    2. 右侧有上升波段（从此低点上升到高点）
    3. 波幅足够大（低点与前后高点的差值占价格的比例达到最小要求）
    4. 在更大的范围内也是真正的谷值（不能被更低点"包住"）
    
    Args:
        low_data: 最低价序列，用于识别波段低点的位置
        close_data: 收盘价序列，用于获取低点的收盘价。如果为None，则使用low_data的值
        min_amplitude_ratio: 最小波幅比例（低点与前后高点的差值占低点价格的比例）
        lookback_window: 向前查找高点的窗口大小
        lookforward_window: 向后查找高点的窗口大小
    
    Returns:
        list of (index, value) tuples，每个元素是一个完整波段的低点
        - index: 低点的位置索引
        - value: 如果提供了close_data，则是该位置的收盘价；否则是low_data的值
    """
    if close_data is None:
        close_data = low_data
    
    troughs = []
    n = len(low_data)
    
    # 使用更大的窗口来检查是否被更低点"包住"
    larger_window = max(lookback_window, lookforward_window) * 2
    
    for i in range(lookback_window, n - lookforward_window):
        current_low = low_data[i]  # 使用最低价来判断是否为波段低点
        current_close = close_data[i]  # 使用收盘价作为返回的值
        
        # 查找左侧最高点（在lookback_window范围内，使用最高价来判断）
        # 注意：这里应该用high_data，但为了简化，我们先用close_data来近似
        # 如果需要更精确，应该传入high_data参数
        left_start = max(0, i - lookback_window)
        left_max_idx = np.argmax(close_data[left_start:i])  # 使用收盘价来查找高点
        left_max_idx = left_start + left_max_idx
        left_max_price = close_data[left_max_idx]
        
        # 查找右侧最高点（在lookforward_window范围内）
        right_end = min(n, i + lookforward_window + 1)
        right_max_idx = np.argmax(close_data[i+1:right_end])
        right_max_idx = i + 1 + right_max_idx
        right_max_price = close_data[right_max_idx]
        
        # 检查是否是一个完整的波段低点：
        # 1. 当前最低价必须低于左侧和右侧的高点
        # 2. 左侧必须有明显的下降（从左侧最高点到当前低点）
        # 3. 右侧必须有明显的上升（从当前低点到右侧最高点）
        # 4. 波幅足够大
        
        left_fall = left_max_price - current_low
        right_rise = right_max_price - current_low
        
        # 计算波幅比例（相对于当前最低价）
        amplitude_ratio = min(left_fall, right_rise) / current_low if current_low > 0 else 0
        
        # 检查是否满足基本条件
        if (current_low < left_max_price and 
            current_low < right_max_price and
            left_fall > 0 and 
            right_rise > 0 and
            amplitude_ratio >= min_amplitude_ratio):
            
            # 额外检查：确保这是一个真正的谷值，而不是上升或下降过程中的点
            # 检查当前点是否在左侧和右侧都是局部最小值
            check_window = min(5, lookback_window // 4, lookforward_window // 4)
            is_local_min = True
            
            # 检查左侧是否有更低的点（使用最低价）
            if i > check_window:
                if min(low_data[i-check_window:i]) <= current_low:
                    is_local_min = False
            
            # 检查右侧是否有更低的点（使用最低价）
            if i < n - check_window:
                if min(low_data[i+1:i+check_window+1]) <= current_low:
                    is_local_min = False
            
            if not is_local_min:
                continue
            
            # 关键检查：在更大的范围内，确保不是被更低点"包住"的点
            # 检查左侧更大范围内是否有更低的点（使用最低价）
            larger_left_start = max(0, i - larger_window)
            larger_left_min = min(low_data[larger_left_start:i]) if i > larger_left_start else current_low
            larger_left_min_idx = larger_left_start + np.argmin(low_data[larger_left_start:i]) if i > larger_left_start else i
            
            # 检查右侧更大范围内是否有更低的点（使用最低价）
            larger_right_end = min(n, i + larger_window + 1)
            larger_right_min = min(low_data[i+1:larger_right_end]) if larger_right_end > i + 1 else current_low
            larger_right_min_idx = i + 1 + np.argmin(low_data[i+1:larger_right_end]) if larger_right_end > i + 1 else i
            
            # 如果左侧和右侧都有更低的点（且差距较大），说明这个点被"包住"了
            # 但只过滤明显被包住的情况（左右都有明显更低的点，且当前点不是趋势的一部分）
            left_lower = larger_left_min < current_low * 0.97  # 3%的容差
            right_lower = larger_right_min < current_low * 0.97
            
            # 如果右侧有更低的点且在较近范围内，说明当前点是下降过程中的点，不是完整波段低点
            if right_lower and larger_right_min_idx < i + lookforward_window:
                # 右侧有更低的点，且距离不远，说明当前点不是完整波段低点
                continue
            
            # 如果左侧有更低的点且在较近范围内，且右侧也有更低的点，说明当前点被包住了
            if left_lower and larger_left_min_idx > i - lookback_window and right_lower:
                # 左右都有更低的点，且都在较近范围内，说明当前点不是完整波段低点
                continue
            
            troughs.append((i, current_close))  # 返回收盘价，而不是最低价
    
    return troughs


def find_uptrend_lines_from_troughs(troughs, min_points=3, min_slope=0.001):
    """
    从局部低点中找到上升趋势线
    根据定义：上升趋势线是在上涨趋势中，连接三个或多个相邻的回调低点所形成的向右上方倾斜的直线
    
    Args:
        troughs: list of (index, value) tuples
        min_points: 趋势线至少需要连接的点数
        min_slope: 最小斜率（正数，表示上升）
    
    Returns:
        list of (slope, intercept, points) tuples，其中points是连接的点的索引列表
    """
    if len(troughs) < min_points:
        return []
    
    # 按索引排序
    troughs_sorted = sorted(troughs, key=lambda x: x[0])
    
    trend_lines = []
    n = len(troughs_sorted)
    
    # 方法1: 优先找相邻的低点组合（更符合定义）
    # 滑动窗口，找到连续的或接近的低点
    for start_idx in range(n - min_points + 1):
        # 尝试从start_idx开始，选择min_points到min_points+2个相邻的低点
        for end_idx in range(start_idx + min_points - 1, min(n, start_idx + min_points + 2)):
            selected_indices = [troughs_sorted[i][0] for i in range(start_idx, end_idx + 1)]
            selected_values = [troughs_sorted[i][1] for i in range(start_idx, end_idx + 1)]
            
            # 检查是否形成上升趋势（值总体上升）
            if len(selected_values) < min_points:
                continue
            
            # 拟合趋势线
            trend_params = fit_trendline_through_points(selected_indices, selected_values)
            if trend_params is None:
                continue
            
            slope, intercept = trend_params
            # 只保留上升趋势（斜率为正）
            if slope > min_slope:
                # 计算R²来评估拟合质量
                y_pred = [slope * idx + intercept for idx in selected_indices]
                ss_res = sum((selected_values[k] - y_pred[k])**2 for k in range(len(selected_values)))
                ss_tot = sum((v - np.mean(selected_values))**2 for v in selected_values)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # 计算点到直线的平均距离
                avg_distance = np.sqrt(ss_res / len(selected_values)) if len(selected_values) > 0 else float('inf')
                min_price = min(selected_values)
                relative_error = avg_distance / min_price if min_price > 0 else 1.0
                
                # 计算相邻性得分：相邻点之间的平均距离越小越好
                if len(selected_indices) > 1:
                    gaps = [selected_indices[i+1] - selected_indices[i] for i in range(len(selected_indices)-1)]
                    avg_gap = np.mean(gaps)
                    max_gap = max(gaps)
                    # 相邻性得分：平均间隔越小、最大间隔越小，得分越高
                    adjacency_score = 1.0 / (1.0 + avg_gap / 50.0 + max_gap / 100.0)
                else:
                    adjacency_score = 0.0
                
                # 放宽条件：R² > 0.7 或相对误差 < 5%
                if r_squared > 0.7 or relative_error < 0.05:
                    # 检查值是否严格递增（更符合定义）
                    is_strictly_increasing = selected_values == sorted(selected_values)
                    strict_increasing_bonus = 0.3 if is_strictly_increasing else 0.0
                    
                    # 连接更低起点的加分：起点值越低，加分越多（更接近最低点）
                    min_all_troughs = min(val for _, val in troughs_sorted) if troughs_sorted else 1.0
                    start_value = selected_values[0] if selected_values else 0
                    # 起点越低，相对于最低点的比例越小，加分越多
                    low_start_bonus = 0.2 * (1.0 - start_value / min_all_troughs) if min_all_troughs > 0 and start_value > 0 else 0.0
                    
                    # 计算综合得分：R² + 相邻性得分 + 严格递增加分 + 低起点加分 - 相对误差
                    composite_score = r_squared + adjacency_score * 0.1 + strict_increasing_bonus + low_start_bonus - relative_error * 0.1
                    trend_lines.append((slope, intercept, selected_indices, r_squared, relative_error, adjacency_score, composite_score, is_strictly_increasing))
    
    # 方法2: 尝试所有可能的组合（确保能找到所有可能的趋势线）
    # 总是执行，因为方法1可能只找到相邻的点，而真正的趋势线可能跳过一些点
    from itertools import combinations
    for combo in combinations(range(n), min_points):
        indices = [troughs_sorted[i][0] for i in combo]
        values = [troughs_sorted[i][1] for i in combo]
        
        if indices != sorted(indices):
            continue
        
        # 检查是否已经存在（避免重复）
        if any(set(indices) == set(existing_indices) for _, _, existing_indices, _, _, _, _, _ in trend_lines):
            continue
        
        trend_params = fit_trendline_through_points(indices, values)
        if trend_params is None:
            continue
        
        slope, intercept = trend_params
        if slope > min_slope:
            y_pred = [slope * idx + intercept for idx in indices]
            ss_res = sum((values[k] - y_pred[k])**2 for k in range(len(values)))
            ss_tot = sum((v - np.mean(values))**2 for v in values)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            avg_distance = np.sqrt(ss_res / len(values)) if len(values) > 0 else float('inf')
            min_price = min(values)
            relative_error = avg_distance / min_price if min_price > 0 else 1.0
            
            if len(indices) > 1:
                gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
                avg_gap = np.mean(gaps)
                max_gap = max(gaps)
                adjacency_score = 1.0 / (1.0 + avg_gap / 50.0 + max_gap / 100.0)
            else:
                adjacency_score = 0.0
            
            if r_squared > 0.7 or relative_error < 0.05:
                # 检查值是否严格递增（更符合定义）
                is_strictly_increasing = values == sorted(values)
                strict_increasing_bonus = 0.3 if is_strictly_increasing else 0.0
                
                # 连接更低起点的加分
                min_all_troughs = min(val for _, val in troughs_sorted) if troughs_sorted else 1.0
                start_value = values[0] if values else 0
                low_start_bonus = 0.2 * (1.0 - start_value / min_all_troughs) if min_all_troughs > 0 and start_value > 0 else 0.0
                
                # 计算综合得分
                composite_score = r_squared + adjacency_score * 0.1 + strict_increasing_bonus + low_start_bonus - relative_error * 0.1
                trend_lines.append((slope, intercept, indices, r_squared, relative_error, adjacency_score, composite_score, is_strictly_increasing))
    
    if not trend_lines:
        return []
    
    # 按综合得分排序：优先保留严格递增、R²高、误差小的
    # 对于严格递增的线，额外提升优先级
    trend_lines.sort(key=lambda x: (x[7], x[6]), reverse=True)  # 先按严格递增排序，再按composite_score排序
    
    # 过滤重叠的趋势线，但优先保留综合得分高的
    filtered_lines = []
    for slope, intercept, indices, r_squared, relative_error, adjacency_score, composite_score, is_strictly_increasing in trend_lines:
        # 检查是否与已有趋势线重叠
        should_add = True
        to_remove = None
        
        for i, (existing_slope, existing_intercept, existing_indices, _, _, _, existing_composite_score, existing_is_strict) in enumerate(filtered_lines):
            # 计算重叠度
            overlap_ratio = len(set(indices) & set(existing_indices)) / min(len(indices), len(existing_indices))
            # 如果重叠度超过50%，检查是否应该替换
            if overlap_ratio > 0.5:
                # 优先保留严格递增的线，如果都是严格递增或都不是，则比较综合得分
                if is_strictly_increasing and not existing_is_strict:
                    to_remove = i
                    should_add = True
                elif not is_strictly_increasing and existing_is_strict:
                    should_add = False
                # 如果都是严格递增，优先保留连接更低起点（更接近最低点）的线
                elif is_strictly_increasing and existing_is_strict:
                    # 比较起点值，保留起点更低的（从peaks_sorted中获取）
                    current_start_value = next((val for idx, val in troughs_sorted if idx == indices[0]), 0) if len(indices) > 0 else 0
                    existing_start_value = next((val for idx, val in troughs_sorted if idx == existing_indices[0]), 0) if len(existing_indices) > 0 else 0
                    if current_start_value < existing_start_value:
                        to_remove = i
                        should_add = True
                    elif current_start_value > existing_start_value:
                        should_add = False
                    # 如果起点相同，比较综合得分
                    elif composite_score > existing_composite_score:
                        to_remove = i
                        should_add = True
                    else:
                        should_add = False
                elif composite_score > existing_composite_score:
                    to_remove = i
                    should_add = True
                else:
                    should_add = False
                break
        
        # 如果需要删除旧线，先删除
        if to_remove is not None:
            filtered_lines.pop(to_remove)
        
        # 添加新线
        if should_add:
            filtered_lines.append((slope, intercept, indices, r_squared, relative_error, adjacency_score, composite_score, is_strictly_increasing))
            # 只保留1条最符合定义的上升趋势线
            if len(filtered_lines) >= 1:
                break
    
    return [(slope, intercept, points) for slope, intercept, points, _, _, _, _, _ in filtered_lines]


def fit_trendline_through_points(indices, values):
    """
    通过给定的点拟合一条直线
    
    Args:
        indices: x坐标（索引）列表
        values: y坐标（价格）列表
    
    Returns:
        (slope, intercept) 或 None
    """
    if len(indices) < 2:
        return None
    # 使用线性回归拟合
    coefs = np.polyfit(indices, values, 1)
    return (coefs[0], coefs[1])


def find_downtrend_lines_from_peaks(peaks, min_points=3, min_slope=-0.001):
    """
    从局部高点中找到下降趋势线
    根据定义：下降趋势线是在下跌趋势中，连接三个或多个相邻的反弹高点所形成的向右下方倾斜的直线
    
    Args:
        peaks: list of (index, value) tuples
        min_points: 趋势线至少需要连接的点数
        min_slope: 最小斜率（负数，表示下降）
    
    Returns:
        list of (slope, intercept, points) tuples，其中points是连接的点的索引列表
    """
    if len(peaks) < min_points:
        return []
    
    # 按索引排序
    peaks_sorted = sorted(peaks, key=lambda x: x[0])
    
    trend_lines = []
    n = len(peaks_sorted)
    
    # 方法1: 优先找相邻的高点组合（更符合定义）
    # 滑动窗口，找到连续的或接近的高点
    for start_idx in range(n - min_points + 1):
        # 尝试从start_idx开始，选择min_points到min_points+2个相邻的高点
        for end_idx in range(start_idx + min_points - 1, min(n, start_idx + min_points + 2)):
            selected_indices = [peaks_sorted[i][0] for i in range(start_idx, end_idx + 1)]
            selected_values = [peaks_sorted[i][1] for i in range(start_idx, end_idx + 1)]
            
            # 检查是否形成下降趋势（值总体下降）
            if len(selected_values) < min_points:
                continue
            
            # 拟合趋势线
            trend_params = fit_trendline_through_points(selected_indices, selected_values)
            if trend_params is None:
                continue
            
            slope, intercept = trend_params
            # 只保留下降趋势（斜率为负）
            if slope < min_slope:
                # 计算R²来评估拟合质量
                y_pred = [slope * idx + intercept for idx in selected_indices]
                ss_res = sum((selected_values[k] - y_pred[k])**2 for k in range(len(selected_values)))
                ss_tot = sum((v - np.mean(selected_values))**2 for v in selected_values)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # 计算点到直线的平均距离
                avg_distance = np.sqrt(ss_res / len(selected_values)) if len(selected_values) > 0 else float('inf')
                max_price = max(selected_values)
                relative_error = avg_distance / max_price if max_price > 0 else 1.0
                
                # 计算相邻性得分：相邻点之间的平均距离越小越好
                if len(selected_indices) > 1:
                    gaps = [selected_indices[i+1] - selected_indices[i] for i in range(len(selected_indices)-1)]
                    avg_gap = np.mean(gaps)
                    max_gap = max(gaps)
                    # 相邻性得分：平均间隔越小、最大间隔越小，得分越高
                    adjacency_score = 1.0 / (1.0 + avg_gap / 50.0 + max_gap / 100.0)
                else:
                    adjacency_score = 0.0
                
                # 放宽条件：R² > 0.7 或相对误差 < 5%
                if r_squared > 0.7 or relative_error < 0.05:
                    # 检查值是否严格递减（更符合定义）
                    is_strictly_decreasing = selected_values == sorted(selected_values, reverse=True)
                    strict_decreasing_bonus = 0.3 if is_strictly_decreasing else 0.0
                    
                    # 连接更高起点的加分：起点值越高，加分越多（更接近最高点）
                    max_all_peaks = max(val for _, val in peaks_sorted) if peaks_sorted else 1.0
                    start_value = selected_values[0] if selected_values else 0
                    high_start_bonus = 0.2 * (start_value / max_all_peaks) if max_all_peaks > 0 else 0.0
                    
                    # 计算综合得分：R² + 相邻性得分 + 严格递减加分 + 高起点加分 - 相对误差
                    composite_score = r_squared + adjacency_score * 0.1 + strict_decreasing_bonus + high_start_bonus - relative_error * 0.1
                    trend_lines.append((slope, intercept, selected_indices, r_squared, relative_error, adjacency_score, composite_score, is_strictly_decreasing))
    
    # 方法2: 尝试所有可能的组合（确保能找到所有可能的趋势线）
    # 总是执行，因为方法1可能只找到相邻的点，而真正的趋势线可能跳过一些点
        from itertools import combinations
        for combo in combinations(range(n), min_points):
            indices = [peaks_sorted[i][0] for i in combo]
            values = [peaks_sorted[i][1] for i in combo]
            
            if indices != sorted(indices):
                continue
            
            # 检查是否已经存在（避免重复）
            if any(set(indices) == set(existing_indices) for _, _, existing_indices, _, _, _, _, _ in trend_lines):
                continue
            
            trend_params = fit_trendline_through_points(indices, values)
            if trend_params is None:
                continue
            
            slope, intercept = trend_params
            if slope < min_slope:
                y_pred = [slope * idx + intercept for idx in indices]
                ss_res = sum((values[k] - y_pred[k])**2 for k in range(len(values)))
                ss_tot = sum((v - np.mean(values))**2 for v in values)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                avg_distance = np.sqrt(ss_res / len(values)) if len(values) > 0 else float('inf')
                max_price = max(values)
                relative_error = avg_distance / max_price if max_price > 0 else 1.0
                
                if len(indices) > 1:
                    gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
                    avg_gap = np.mean(gaps)
                    max_gap = max(gaps)
                    adjacency_score = 1.0 / (1.0 + avg_gap / 50.0 + max_gap / 100.0)
                else:
                    adjacency_score = 0.0
                
                if r_squared > 0.7 or relative_error < 0.05:
                    # 检查值是否严格递减（更符合定义）
                    is_strictly_decreasing = values == sorted(values, reverse=True)
                    strict_decreasing_bonus = 0.3 if is_strictly_decreasing else 0.0
                    
                    # 连接更高起点的加分：起点值越高，加分越多（更接近最高点）
                    max_all_peaks = max(val for _, val in peaks_sorted) if peaks_sorted else 1.0
                    start_value = values[0] if values else 0
                    high_start_bonus = 0.2 * (start_value / max_all_peaks) if max_all_peaks > 0 else 0.0
                    
                    # 计算综合得分：R² + 相邻性得分 + 严格递减加分 + 高起点加分 - 相对误差
                    composite_score = r_squared + adjacency_score * 0.1 + strict_decreasing_bonus + high_start_bonus - relative_error * 0.1
                    trend_lines.append((slope, intercept, indices, r_squared, relative_error, adjacency_score, composite_score, is_strictly_decreasing))
    
    if not trend_lines:
        return []
    
    # 按综合得分排序：优先保留严格递减、R²高、误差小的
    # 对于严格递减的线，额外提升优先级
    trend_lines.sort(key=lambda x: (x[7], x[6]), reverse=True)  # 先按严格递减排序，再按composite_score排序
    
    # 过滤重叠的趋势线，但优先保留综合得分高的
    filtered_lines = []
    for slope, intercept, indices, r_squared, relative_error, adjacency_score, composite_score, is_strictly_decreasing in trend_lines:
        # 检查是否与已有趋势线重叠
        should_add = True
        to_remove = None
        
        for i, (existing_slope, existing_intercept, existing_indices, _, _, _, existing_composite_score, existing_is_strict) in enumerate(filtered_lines):
            # 计算重叠度
            overlap_ratio = len(set(indices) & set(existing_indices)) / min(len(indices), len(existing_indices))
            # 如果重叠度超过50%，检查是否应该替换
            if overlap_ratio > 0.5:
                # 优先保留严格递减的线
                if is_strictly_decreasing and not existing_is_strict:
                    to_remove = i
                    should_add = True
                elif not is_strictly_decreasing and existing_is_strict:
                    should_add = False
                # 如果都是严格递减，优先保留连接更高起点（更接近最高点）的线
                elif is_strictly_decreasing and existing_is_strict:
                    # 比较起点值，保留起点更高的（从peaks_sorted中获取）
                    current_start_value = next((val for idx, val in peaks_sorted if idx == indices[0]), 0) if len(indices) > 0 else 0
                    existing_start_value = next((val for idx, val in peaks_sorted if idx == existing_indices[0]), 0) if len(existing_indices) > 0 else 0
                    if current_start_value > existing_start_value:
                        to_remove = i
                        should_add = True
                    elif current_start_value < existing_start_value:
                        should_add = False
                    # 如果起点相同，比较综合得分
                    elif composite_score > existing_composite_score:
                        to_remove = i
                        should_add = True
                    else:
                        should_add = False
                elif composite_score > existing_composite_score:
                    to_remove = i
                    should_add = True
                else:
                    should_add = False
                break
        
        # 如果需要删除旧线，先删除
        if to_remove is not None:
            filtered_lines.pop(to_remove)
        
        # 添加新线
        if should_add:
            filtered_lines.append((slope, intercept, indices, r_squared, relative_error, adjacency_score, composite_score, is_strictly_decreasing))
            # 只保留1条最符合定义的下降趋势线（优先保留连接更高起点、斜率更陡的严格递减线）
            if len(filtered_lines) >= 1:
                break
    
    return [(slope, intercept, points) for slope, intercept, points, _, _, _, _, _ in filtered_lines]


def get_line_points(candles, line_points):
    # Place line points in tuples for matplotlib finance
    # https://github.com/matplotlib/mplfinance/blob/master/examples/using_lines.ipynb
    idx = candles.index
    line_i = len(candles) - len(line_points)
    assert line_i >= 0
    points = []
    for i in range(line_i, len(candles)):
        points.append((idx[i], line_points[i - line_i]))
    return points


def split_line_into_segments(line_points):
    return [[line_points[i], line_points[i + 1]] for i in range(len(line_points) - 1)]


# Calculate MACD using TA-Lib
# Typical parameters: fastperiod=12, slowperiod=26, signalperiod=9


class TechnicalTools:

    @staticmethod
    @tool
    def generate_trend_image(
        kline_data: Annotated[
            dict,
            "Dictionary containing OHLCV data with keys 'Datetime', 'Open', 'High', 'Low', 'Close'.",
        ]
    ) -> dict:
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
                print(f"[DEBUG] 找到 {len(downtrend_lines)} 条下降趋势线（基于完整波段高点）")
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
            "trend_graph.png",
            format="png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.1,
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

    @staticmethod
    @tool
    def generate_kline_image(
        kline_data: Annotated[
            dict,
            "Dictionary containing OHLCV data with keys 'Datetime', 'Open', 'High', 'Low', 'Close'.",
        ],
    ) -> dict:
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

    @staticmethod
    @tool
    def compute_rsi(
        kline_data: Annotated[
            dict,
            "Dictionary with a 'Close' key containing a list of float closing prices. NOTE: This parameter is automatically injected by the system - you do NOT need to provide it in tool calls. Just call the tool with the period parameter if needed.",
        ],
        period: Annotated[
            int, "Lookback period for RSI calculation (default is 14)"
        ] = 14,
    ) -> dict:
        """
        Compute the Relative Strength Index (RSI) using TA-Lib.

        Args:
            data (dict): Dictionary containing at least a 'Close' key with a list of float values.
            period (int): Lookback period for RSI calculation (default is 14).

        Returns:
            dict: A dictionary with a single key 'rsi' mapping to a list of RSI values.
        """
        df = pd.DataFrame(kline_data)
        rsi = talib.RSI(df["Close"], timeperiod=period)
        return {"rsi": rsi.fillna(0).round(2).tolist()[-28:]}

    @staticmethod
    @tool
    def compute_macd(
        kline_data: Annotated[
            dict,
            "Dictionary with a 'Close' key containing a list of float closing prices. NOTE: This parameter is automatically injected by the system - you do NOT need to provide it in tool calls. Just call the tool with fastperiod, slowperiod, and signalperiod parameters if needed.",
        ],
        fastperiod: Annotated[int, "Fast EMA period"] = 12,
        slowperiod: Annotated[int, "Slow EMA period"] = 26,
        signalperiod: Annotated[int, "Signal line EMA period"] = 9,
    ) -> dict:
        """
        Compute the Moving Average Convergence Divergence (MACD) using TA-Lib.

        Args:
            kline_data (dict): Dictionary containing a 'Close' key with list of float values.
            fastperiod (int): Fast EMA period.
            slowperiod (int): Slow EMA period.
            signalperiod (int): Signal line EMA period.

        Returns:
            dict: Dictionary containing 'macd', 'macd_signal', and 'macd_hist' as lists of values.
        """
        df = pd.DataFrame(kline_data)
        macd, macd_signal, macd_hist = talib.MACD(
            df["Close"],
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod,
        )
        return {
            "macd": macd.fillna(0).round(2).tolist(),
            "macd_signal": macd_signal.fillna(0).round(2).tolist()[-28:],
            "macd_hist": macd_hist.fillna(0).round(2).tolist()[-28:],
        }

    @staticmethod
    @tool
    def compute_stoch(
        kline_data: Annotated[
            dict,
            "Dictionary with 'High', 'Low', and 'Close' keys, each mapping to lists of float values. NOTE: This parameter is automatically injected by the system - you do NOT need to provide it in tool calls. Just call the tool without any arguments.",
        ]
    ) -> dict:
        """
        Compute the Stochastic Oscillator %K and %D using TA-Lib.

        Args:
            kline_data (dict): Dictionary with 'High', 'Low', and 'Close' keys, each mapping to lists of float values.

        Returns:
            dict: A dictionary with keys 'stoch_k' and 'stoch_d',
                each mapping to a list representing %K and %D values.
        """
        df = pd.DataFrame(kline_data)
        stoch_k, stoch_d = talib.STOCH(
            df["High"],
            df["Low"],
            df["Close"],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3,
        )
        return {
            "stoch_k": stoch_k.fillna(0).round(2).tolist()[-28:],
            "stoch_d": stoch_d.fillna(0).round(2).tolist()[-28:],
        }

    @staticmethod
    @tool
    def compute_roc(
        kline_data: Annotated[
            dict,
            "Dictionary with a 'Close' key containing a list of float closing prices. NOTE: This parameter is automatically injected by the system - you do NOT need to provide it in tool calls. Just call the tool with the period parameter if needed.",
        ],
        period: Annotated[
            int, "Number of periods over which to calculate ROC (default is 10)"
        ] = 10,
    ) -> dict:
        """
        Compute the Rate of Change (ROC) indicator using TA-Lib.

        Args:
            kline_data (dict): Dictionary containing a 'Close' key with a list of float values.
            period (int): Number of periods over which to calculate ROC (default is 10).

        Returns:
            dict: A dictionary with a single key 'roc' mapping to a list of ROC values.
        """

        df = pd.DataFrame(kline_data)
        roc = talib.ROC(df["Close"], timeperiod=period)
        return {"roc": roc.fillna(0).round(2).tolist()[-28:]}

    @staticmethod
    @tool
    def compute_willr(
        kline_data: Annotated[
            dict,
            "Dictionary with 'High', 'Low', and 'Close' keys containing float lists. NOTE: This parameter is automatically injected by the system - you do NOT need to provide it in tool calls. Just call the tool with the period parameter if needed.",
        ],
        period: Annotated[int, "Lookback period for Williams %R"] = 14,
    ) -> dict:
        """
        Compute the Williams %R indicator using TA-Lib.

        Args:
            kline_data (dict): Dictionary with 'High', 'Low', and 'Close' keys.
            period (int): Lookback period for Williams %R calculation.

        Returns:
            dict: Dictionary with key 'willr' mapping to the list of Williams %R values.
        """
        # print("-------------------------CALLED COMPUTE WILLR--------------------------\n")
        df = pd.DataFrame(kline_data)
        willr = talib.WILLR(df["High"], df["Low"], df["Close"], timeperiod=period)
        return {"willr": willr.fillna(0).round(2).tolist()[-28:]}
