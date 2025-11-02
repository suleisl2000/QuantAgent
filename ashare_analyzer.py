#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股股票技术分析工具
使用多智能体系统对A股进行技术指标、形态和趋势分析

主要功能:
    - 从akshare获取A股历史数据（支持前复权）
    - 生成K线图和趋势图
    - 使用多智能体系统进行技术分析（指标、形态、趋势、决策）
    - 支持多种时间周期（1分钟到1月）
    - 支持指定日期范围或天数

使用方法:
    python ashare_analyzer.py                    # 使用默认参数（600036 招商银行，1d周期）
    python ashare_analyzer.py --code 600519      # 指定股票代码
    python ashare_analyzer.py --code 600519 --interval 1d  # 指定代码和时间周期
    python ashare_analyzer.py --code 601698 --start-date 20250801 --end-date 20251102  # 指定日期范围
    python ashare_analyzer.py --code 601698 --data-only  # 仅获取数据和生成图表
"""

import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import pandas as pd
import akshare as ak
from trading_graph import TradingGraph
import static_util


def fetch_akshare_data(symbol: str, interval: str, start_datetime: datetime, end_datetime: datetime) -> pd.DataFrame:
    """
    从akshare获取A股OHLCV数据
    
    Args:
        symbol: 6位A股代码，如 "600036"
        interval: 时间间隔，如 "1h", "4h", "1d", "15m"
        start_datetime: 开始时间
        end_datetime: 结束时间
    
    Returns:
        DataFrame包含Datetime, Open, High, Low, Close, Volume列
    """
    # 清理代码（确保是6位数字）
    clean_symbol = symbol.replace(".SH", "").replace(".SZ", "").strip()
    
    if not clean_symbol.isdigit() or len(clean_symbol) != 6:
        raise ValueError(f"Invalid stock symbol format: {symbol}, expected 6-digit code")
    
    # akshare接口映射
    akshare_interface_map = {
        # 分钟数据使用 stock_zh_a_hist_min_em
        "1m": {"func": "stock_zh_a_hist_min_em", "period": "1"},
        "5m": {"func": "stock_zh_a_hist_min_em", "period": "5"},
        "15m": {"func": "stock_zh_a_hist_min_em", "period": "15"},
        "30m": {"func": "stock_zh_a_hist_min_em", "period": "30"},
        "60m": {"func": "stock_zh_a_hist_min_em", "period": "60"},
        "1h": {"func": "stock_zh_a_hist_min_em", "period": "60"},  # 1小时=60分钟
        # 日/周/月数据使用 stock_zh_a_hist
        "1d": {"func": "stock_zh_a_hist", "period": "daily"},
        "1w": {"func": "stock_zh_a_hist", "period": "weekly"},
        "1mo": {"func": "stock_zh_a_hist", "period": "monthly"},
    }
    
    interface_config = akshare_interface_map.get(interval)
    if not interface_config:
        raise ValueError(f"Unsupported interval: {interval}")
    
    func_name = interface_config["func"]
    period = interface_config["period"]
    
    # 格式化日期（akshare需要YYYYMMDD格式）
    start_date_str = start_datetime.strftime("%Y%m%d")
    end_date_str = end_datetime.strftime("%Y%m%d")
    
    print(f"正在获取数据: {symbol}, 周期: {interval}, 时间范围: {start_date_str} 至 {end_date_str}")
    
    # 调用对应的akshare函数
    if func_name == "stock_zh_a_hist_min_em":
        df = ak.stock_zh_a_hist_min_em(
            symbol=clean_symbol,
            period=period,
            start_date=start_date_str,
            end_date=end_date_str,
            adjust="qfq",  # 前复权
        )
    else:  # stock_zh_a_hist
        df = ak.stock_zh_a_hist(
            symbol=clean_symbol,
            period=period,
            start_date=start_date_str,
            end_date=end_date_str,
            adjust="qfq",  # 前复权
        )
    
    if df is None or df.empty:
        raise ValueError(f"无法获取数据: {symbol}")
    
    # akshare返回中文列名，需要映射为英文
    column_mapping = {
        "时间": "Datetime",  # 分钟数据使用"时间"
        "日期": "Datetime",  # 日/周/月数据使用"日期"
        "开盘": "Open",
        "收盘": "Close",
        "最高": "High",
        "最低": "Low",
        "成交量": "Volume",
    }
    
    # 重命名列（仅重命名存在的列）
    existing_mapping = {old: new for old, new in column_mapping.items() if old in df.columns}
    df = df.rename(columns=existing_mapping)
    
    # 确保必需的列存在
    required_columns = ["Datetime", "Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"缺少必需的列。可用列: {list(df.columns)}")
    
    # 转换为datetime并排序
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df[required_columns].sort_values("Datetime").reset_index(drop=True)
    
    print(f"成功获取 {len(df)} 条数据")
    print(f"日期范围: {df['Datetime'].min()} 至 {df['Datetime'].max()}")
    
    return df


def prepare_data_for_analysis(df: pd.DataFrame) -> dict:
    """
    准备数据用于分析
    将DataFrame转换为TradingGraph需要的字典格式
    
    Args:
        df: 包含OHLCV数据的DataFrame
    
    Returns:
        字典格式的K线数据
    """
    # 使用所有数据，不裁剪 - 数据量由days参数控制
    df_slice = df.copy()
    
    required_columns = ["Datetime", "Open", "High", "Low", "Close"]
    df_slice = df_slice[required_columns].reset_index(drop=True)
    
    # 转换为字典格式
    df_slice_dict = {}
    for col in required_columns:
        if col == "Datetime":
            # 将datetime对象转换为字符串以便JSON序列化
            df_slice_dict[col] = df_slice[col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        else:
            df_slice_dict[col] = df_slice[col].tolist()
    
    return df_slice_dict


def format_timeframe(timeframe: str) -> str:
    """
    格式化时间框架用于显示
    """
    display_timeframe = timeframe
    if timeframe.endswith("h"):
        display_timeframe += "our"
    elif timeframe.endswith("m"):
        display_timeframe += "in"
    elif timeframe.endswith("d"):
        display_timeframe += "ay"
    elif timeframe == "1w":
        display_timeframe = "1 week"
    elif timeframe == "1mo":
        display_timeframe = "1 month"
    return display_timeframe


def analyze_ashare(stock_code: str = "600036", interval: str = "1h", days: int = 30, 
                         start_date: str = None, end_date: str = None, data_only: bool = False):
    """
    A股技术分析主函数
    
    Args:
        stock_code: 6位A股代码，如 "600036"
        interval: 时间周期，如 "1h", "4h", "1d", "15m"
        days: 获取多少天的历史数据（当start_date和end_date未指定时生效）
        start_date: 起始日期，格式 yyyymmdd (如 "20250801")，如果指定则days参数不生效
        end_date: 结束日期，格式 yyyymmdd (如 "20251102")，如果指定则days参数不生效
        data_only: 如果为True，仅获取数据并生成图表后退出，不进行智能体分析
    """
    print("=" * 60)
    print("A股股票技术分析")
    print("=" * 60)
    
    # A股代码到名称的映射（部分常用股票）
    stock_name_map = {
        "000001": "平安银行",
        "000002": "万科A",
        "600000": "浦发银行",
        "600036": "招商银行",
        "600519": "贵州茅台",
        "000858": "五粮液",
        "002415": "海康威视",
        "000063": "中兴通讯",
        "600887": "伊利股份",
        "000776": "广发证券",
    }
    
    stock_name = stock_name_map.get(stock_code, stock_code)
    
    # 创建输出目录（带时间戳）- 在开始时创建，确保后续步骤都能使用
    output_dir = Path(f"analysis_output_{stock_code}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(exist_ok=True)
    
    # 计算时间范围
    if start_date and end_date:
        # 使用指定的起始和结束日期
        try:
            start_datetime = datetime.strptime(start_date, "%Y%m%d")
            end_datetime = datetime.strptime(end_date, "%Y%m%d")
            
            # 验证日期顺序
            if start_datetime >= end_datetime:
                print(f"\n错误: 起始日期 ({start_date}) 必须早于结束日期 ({end_date})")
                return
            
            # 结束日期不能是未来
            if end_datetime > datetime.now():
                print(f"\n警告: 结束日期 ({end_date}) 是未来日期，将使用当前时间")
                end_datetime = datetime.now()
                
            print(f"\n股票代码: {stock_code} ({stock_name})")
            print(f"时间周期: {interval}")
            print(f"时间范围: {start_datetime.strftime('%Y-%m-%d')} 至 {end_datetime.strftime('%Y-%m-%d')} (指定日期)")
            
        except ValueError as e:
            print(f"\n错误: 日期格式错误 - {e}")
            print("日期格式应为: yyyymmdd (例如: 20250801)")
            return
    else:
        # 使用days参数计算时间范围
        end_datetime = datetime.now()
        start_datetime = end_datetime - timedelta(days=days)
        
        print(f"\n股票代码: {stock_code} ({stock_name})")
        print(f"时间周期: {interval}")
        print(f"时间范围: {start_datetime.strftime('%Y-%m-%d')} 至 {end_datetime.strftime('%Y-%m-%d')} (最近{days}天)")
    
    print(f"输出目录: {output_dir.absolute()}")
    
    # 2. 检查API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n警告: 未设置 OPENAI_API_KEY 环境变量")
        print("请运行: export OPENAI_API_KEY='your_api_key_here'")
        print("注意: 如果使用DashScope，请将DashScope API Key设置为OPENAI_API_KEY")
        return
    
    # 检查当前使用的LLM配置
    from default_config import DEFAULT_CONFIG
    base_url = DEFAULT_CONFIG.get("base_url")
    agent_model = DEFAULT_CONFIG.get("agent_llm_model", "gpt-4o-mini")
    graph_model = DEFAULT_CONFIG.get("graph_llm_model", "gpt-4o")
    
    print(f"\nAPI密钥: {api_key[:10]}... (已设置)")
    if base_url:
        print(f"LLM提供商: DashScope (Tongyi Qianwen)")
        print(f"  智能体模型: {agent_model}")
        print(f"  图表模型: {graph_model}")
    else:
        print(f"LLM提供商: OpenAI")
        print(f"  智能体模型: {agent_model}")
        print(f"  图表模型: {graph_model}")
    
    # 3. 获取数据
    try:
        print("\n" + "-" * 60)
        print("步骤1: 获取市场数据")
        print("-" * 60)
        df = fetch_akshare_data(stock_code, interval, start_datetime, end_datetime)
        
        if df.empty:
            print("错误: 未获取到数据")
            return
        
        print(f"\n完整数据 ({len(df)} 条记录):")
        print("=" * 60)
        # 设置pandas显示选项，显示所有数据
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(df)
        print("=" * 60)
        
        print(f"\n数据统计:")
        print(df.describe())
        
    except Exception as e:
        print(f"错误: 获取数据失败 - {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 准备数据
    try:
        print("\n" + "-" * 60)
        print("步骤2: 准备分析数据")
        print("-" * 60)
        kline_data = prepare_data_for_analysis(df)
        print(f"已准备 {len(kline_data['Datetime'])} 根K线数据")
        
    except Exception as e:
        print(f"错误: 数据准备失败 - {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 生成图表
    pattern_image_path = None
    trend_image_path = None
    try:
        print("\n" + "-" * 60)
        print("步骤3: 生成K线图表")
        print("-" * 60)
        pattern_image = static_util.generate_kline_image(kline_data)
        trend_image = static_util.generate_trend_image(kline_data)
        
        # 复制图片到输出目录
        pattern_image_path = output_dir / "kline_chart.png"
        trend_image_path = output_dir / "trend_graph.png"
        
        if os.path.exists("kline_chart.png"):
            shutil.copy("kline_chart.png", pattern_image_path)
            print(f"K线图表已保存到: {pattern_image_path.absolute()}")
        if os.path.exists("trend_graph.png"):
            shutil.copy("trend_graph.png", trend_image_path)
            print(f"趋势图表已保存到: {trend_image_path.absolute()}")
        
        print("图表生成成功")
        
    except Exception as e:
        print(f"警告: 图表生成失败 - {e}")
        pattern_image = {"pattern_image": None}
        trend_image = {"trend_image": None}
    
    # 如果设置了data_only参数，只获取数据和生成图表后退出
    if data_only:
        print("\n" + "=" * 60)
        print("数据获取和图表生成完成（已启用 --data-only 模式，跳过分析步骤）")
        print("=" * 60)
        return
    
    # 6. 初始化TradingGraph
    try:
        print("\n" + "-" * 60)
        print("步骤4: 初始化交易分析图")
        print("-" * 60)
        
        # 可以自定义配置（可选）
        # from default_config import DEFAULT_CONFIG
        # config = DEFAULT_CONFIG.copy()
        # config["agent_llm_model"] = "gpt-4o-mini"
        # config["graph_llm_model"] = "gpt-4o"
        # trading_graph = TradingGraph(config=config)
        
        trading_graph = TradingGraph()
        print("TradingGraph 初始化成功")
        
    except Exception as e:
        print(f"错误: TradingGraph初始化失败 - {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 创建初始状态
    display_timeframe = format_timeframe(interval)
    initial_state = {
        "kline_data": kline_data,
        "analysis_results": None,
        "messages": [],
        "time_frame": display_timeframe,
        "stock_name": stock_name,
        "pattern_image": pattern_image.get("pattern_image"),
        "trend_image": trend_image.get("trend_image"),
    }
    
    # 8. 运行分析
    try:
        print("\n" + "-" * 60)
        print("步骤5: 运行多智能体分析")
        print("-" * 60)
        print("这可能需要几分钟时间，请耐心等待...")
        
        final_state = trading_graph.graph.invoke(initial_state)
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)
        
        # 计算数据的时间范围
        data_start_time = df['Datetime'].min()
        data_end_time = df['Datetime'].max()
        
        # 9. 输出结果
        print("\n" + "-" * 60)
        print("分析结果")
        print("-" * 60)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"分析数据时间段: {data_start_time.strftime('%Y-%m-%d %H:%M:%S')} 至 {data_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 指标报告
        if "indicator_report" in final_state:
            print("\n【技术指标分析】")
            print(final_state["indicator_report"])
        
        # 模式识别报告
        if "pattern_report" in final_state:
            print("\n【K线形态识别】")
            print(final_state["pattern_report"])
        
        # 趋势分析报告
        if "trend_report" in final_state:
            print("\n【趋势分析】")
            print(final_state["trend_report"])
        
        # 最终交易决策
        if "final_trade_decision" in final_state:
            print("\n【最终交易决策】")
            print(final_state["final_trade_decision"])
        
        # 保存完整结果到文件
        result_file = output_dir / f"analysis_result_{stock_code}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # 使用已计算的时间范围
        analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"A股分析结果\n")
            f.write(f"{'='*60}\n")
            f.write(f"股票代码: {stock_code} ({stock_name})\n")
            f.write(f"时间周期: {display_timeframe}\n")
            f.write(f"分析时间: {analysis_time}\n")
            f.write(f"分析数据时间段: {data_start_time.strftime('%Y-%m-%d %H:%M:%S')} 至 {data_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            
            if "indicator_report" in final_state:
                f.write("【技术指标分析】\n")
                f.write(str(final_state["indicator_report"]))
                f.write("\n\n")
            
            if "pattern_report" in final_state:
                f.write("【K线形态识别】\n")
                f.write(str(final_state["pattern_report"]))
                f.write("\n\n")
            
            if "trend_report" in final_state:
                f.write("【趋势分析】\n")
                f.write(str(final_state["trend_report"]))
                f.write("\n\n")
            
            if "final_trade_decision" in final_state:
                f.write("【最终交易决策】\n")
                f.write(str(final_state["final_trade_decision"]))
                f.write("\n\n")
            
            # 在文件中也记录图片保存位置
            f.write(f"{'='*60}\n")
            f.write("生成的文件:\n")
            f.write(f"分析结果文件: {result_file.name}\n")
            if pattern_image_path and pattern_image_path.exists():
                f.write(f"K线图表: {pattern_image_path.name}\n")
            if trend_image_path and trend_image_path.exists():
                f.write(f"趋势图表: {trend_image_path.name}\n")
            f.write(f"所有文件保存在目录: {output_dir.absolute()}\n")
        
        print(f"\n完整结果已保存到: {result_file.absolute()}")
        
        # 打印图片保存位置
        if pattern_image_path and pattern_image_path.exists():
            print(f"K线图表已保存到: {pattern_image_path.absolute()}")
        if trend_image_path and trend_image_path.exists():
            print(f"趋势图表已保存到: {trend_image_path.absolute()}")
        
    except Exception as e:
        print(f"\n错误: 分析过程失败 - {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A股股票技术分析工具 - 使用多智能体系统进行技术分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python ashare_analyzer.py                           # 使用默认参数
  python ashare_analyzer.py --code 600519             # 分析贵州茅台
  python ashare_analyzer.py --code 000001 --interval 1d  # 分析平安银行，日K线
  python ashare_analyzer.py --code 600036 --interval 15m --days 7  # 15分钟K线，最近7天
  python ashare_analyzer.py --code 601698 --start-date 20250801 --end-date 20251102  # 指定日期范围
  python ashare_analyzer.py --code 601698 --data-only  # 仅获取数据和生成图表，不进行分析
        """
    )
    parser.add_argument(
        "--code",
        type=str,
        default="600036",
        help="6位A股代码 (默认: 600036 招商银行)"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=["1m", "5m", "15m", "30m", "60m", "1h", "1d", "1w", "1mo"],
        help="时间周期 (默认: 1d)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="获取多少天的历史数据 (默认: 90，当指定了--start-date和--end-date时不生效)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="起始日期，格式: yyyymmdd (例如: 20250801)。如果指定此参数和--end-date，--days参数不生效"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="结束日期，格式: yyyymmdd (例如: 20251102)。如果指定此参数和--start-date，--days参数不生效"
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="仅获取数据和生成图表，执行到步骤3后退出，不进行智能体分析"
    )
    
    args = parser.parse_args()
    
    # 验证日期参数：要么都指定，要么都不指定
    if (args.start_date is None) != (args.end_date is None):
        parser.error("--start-date 和 --end-date 必须同时指定或都不指定")
    
    analyze_ashare(
        stock_code=args.code,
        interval=args.interval,
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
        data_only=args.data_only
    )

