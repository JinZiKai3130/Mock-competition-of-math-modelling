import pandas as pd
import numpy as np
from datetime import datetime, time

# 读取工作日和周末的数据
weekday_df = pd.read_csv('MCM2026 Training Test Problem B/hall_calls_5min_weekday_avg.csv')
weekend_df = pd.read_csv('MCM2026 Training Test Problem B/hall_calls_5min_weekend_avg.csv')

# 将时间列转换为datetime.time类型以便比较
def convert_time_str_to_time(time_str):
    """将时间字符串转换为datetime.time对象"""
    return datetime.strptime(time_str, '%H:%M').time()

# 为两个DataFrame添加时间对象列
weekday_df['time_obj'] = weekday_df['time_of_day'].apply(convert_time_str_to_time)
weekend_df['time_obj'] = weekend_df['time_of_day'].apply(convert_time_str_to_time)

# 定义基于客流量的交通模式分配函数
def assign_traffic_pattern_by_volume(row):
    """根据客流量特征分配交通模式"""
    total_calls = row['总次数']
    floor1_calls = row['楼层1']
    floor14_calls = row.get('楼层14', 0)  # 使用get防止列不存在
    
    current_time = row['time_obj']
    
    # 定义时间边界
    morning_start = time(7, 0)
    morning_end = time(9, 30)
    
    breakfast_start = time(9, 30)
    breakfast_end = time(11, 0)
    
    lunch_start = time(11, 0)
    lunch_end = time(13, 30)
    
    afternoon_start = time(13, 30)
    afternoon_end = time(17, 0)
    
    evening_peak_start = time(17, 0)
    evening_peak_end = time(18, 0)
    
    dinner_start = time(18, 0)
    dinner_end = time(20, 0)
    
    evening_leisure_start = time(20, 0)
    evening_leisure_end = time(22, 0)
    
    night_start = time(22, 0)
    night_end = time(7, 0)
    
    # 夜间时段（跨午夜）
    if current_time >= night_start or current_time < morning_start:
        return 'Night/Overnight'
    
    # Morning Up-Peak: 早上7:00-9:30，且1楼客流量占比高
    elif morning_start <= current_time < morning_end:
        if floor1_calls > total_calls * 0.3:  # 1楼占比超过30%
            return 'Morning Up-Peak'
        else:
            return 'Morning Inter-floor'
    
    # Breakfast Hour: 9:30-11:00，总客流量较高
    elif breakfast_start <= current_time < breakfast_end:
        if total_calls > 15:  # 周末可能阈值较低
            return 'Breakfast Hour'
        else:
            return 'Morning Inter-floor'
    
    # Lunch Hour: 11:00-13:30，总客流量高，且分布相对均匀
    elif lunch_start <= current_time < lunch_end:
        if total_calls > 20:  # 周末可能阈值较低
            return 'Lunch Hour'
        else:
            return 'Afternoon Inter-floor'
    
    # Meeting/Inter-floor traffic: 13:30-17:00
    elif afternoon_start <= current_time < afternoon_end:
        return 'Meeting/Inter-floor'
    
    # Evening Down-Peak: 17:00-18:00，总客流量中等
    elif evening_peak_start <= current_time < evening_peak_end:
        return 'Evening Down-Peak'
    
    # Dinner Hour: 18:00-20:00，总客流量非常高，且14楼客流量高
    elif dinner_start <= current_time < dinner_end:
        if total_calls > 40 and floor14_calls > 3:  # 周末可能阈值较低
            return 'Dinner Hour'
        else:
            return 'Evening Leisure'
    
    # Evening Leisure: 20:00-22:00
    elif evening_leisure_start <= current_time < evening_leisure_end:
        return 'Evening Leisure'
    
    else:
        return 'Night/Overnight'

# 为工作日和周末数据分配交通模式
weekday_df['traffic_pattern'] = weekday_df.apply(assign_traffic_pattern_by_volume, axis=1)
weekend_df['traffic_pattern'] = weekend_df.apply(assign_traffic_pattern_by_volume, axis=1)

# 计算每个交通模式的统计信息
def calculate_pattern_stats(df, day_type):
    """计算每个交通模式的统计信息"""
    patterns = df['traffic_pattern'].unique()
    pattern_stats = []
    
    for pattern in patterns:
        pattern_data = df[df['traffic_pattern'] == pattern]
        
        # 基本统计
        stats = {
            'Day_Type': day_type,
            'Traffic_Pattern': pattern,
            'Num_Time_Slots': len(pattern_data),
            'Time_Range': f"{pattern_data['time_of_day'].min()} - {pattern_data['time_of_day'].max()}",
            'Avg_Total_Calls': pattern_data['总次数'].mean(),
            'Max_Total_Calls': pattern_data['总次数'].max(),
            'Min_Total_Calls': pattern_data['总次数'].min(),
            'Std_Total_Calls': pattern_data['总次数'].std(),
            'Avg_Floor1_Calls': pattern_data['楼层1'].mean(),
            'Avg_Floor14_Calls': pattern_data.get('楼层14', pd.Series([0])).mean() if '楼层14' in pattern_data.columns else 0,
            'Floor1_Ratio': (pattern_data['楼层1'].sum() / pattern_data['总次数'].sum()) if pattern_data['总次数'].sum() > 0 else 0,
        }
        
        if '楼层14' in pattern_data.columns:
            stats['Floor14_Ratio'] = (pattern_data['楼层14'].sum() / pattern_data['总次数'].sum()) if pattern_data['总次数'].sum() > 0 else 0
        
        # 计算每个楼层的平均呼叫次数
        floor_cols = [col for col in df.columns if col.startswith('楼层')]
        for floor_col in floor_cols:
            stats[f'Avg_{floor_col}'] = pattern_data[floor_col].mean()
        
        pattern_stats.append(stats)
    
    return pd.DataFrame(pattern_stats)

# 计算工作日和周末的统计信息
weekday_stats = calculate_pattern_stats(weekday_df, 'Weekday')
weekend_stats = calculate_pattern_stats(weekend_df, 'Weekend')

# 合并统计信息
all_stats = pd.concat([weekday_stats, weekend_stats], ignore_index=True)

# 保存结果到CSV文件
all_stats.to_csv('MCM2026 Training Test Problem B/traffic_patterns_volume_based_stats.csv', 
                 index=False, encoding='utf-8-sig')

# 保存带交通模式的详细数据
weekday_df[['time_of_day', '总次数', '楼层1', '楼层14', 'traffic_pattern']].to_csv(
    'MCM2026 Training Test Problem B/weekday_traffic_patterns_detailed.csv', 
    index=False, encoding='utf-8-sig')

weekend_df[['time_of_day', '总次数', '楼层1', '楼层14', 'traffic_pattern']].to_csv(
    'MCM2026 Training Test Problem B/weekend_traffic_patterns_detailed.csv', 
    index=False, encoding='utf-8-sig')

print("交通模式划分完成！基于客流量的分配结果已保存。")
print(f"工作日数据条数: {len(weekday_df)}")
print(f"周末数据条数: {len(weekend_df)}")

# 打印工作日交通模式分布
print("\n工作日交通模式分布:")
weekday_counts = weekday_df['traffic_pattern'].value_counts()
for pattern, count in weekday_counts.items():
    percentage = count / len(weekday_df) * 100
    time_range = weekday_df[weekday_df['traffic_pattern'] == pattern]['time_of_day']
    avg_calls = weekday_df[weekday_df['traffic_pattern'] == pattern]['总次数'].mean()
    print(f"  {pattern}: {count}个时间段 ({percentage:.1f}%) | 时间: {time_range.min()} - {time_range.max()} | 平均客流量: {avg_calls:.2f}")

print("\n周末交通模式分布:")
weekend_counts = weekend_df['traffic_pattern'].value_counts()
for pattern, count in weekend_counts.items():
    percentage = count / len(weekend_df) * 100
    time_range = weekend_df[weekend_df['traffic_pattern'] == pattern]['time_of_day']
    avg_calls = weekend_df[weekend_df['traffic_pattern'] == pattern]['总次数'].mean()
    print(f"  {pattern}: {count}个时间段 ({percentage:.1f}%) | 时间: {time_range.min()} - {time_range.max()} | 平均客流量: {avg_calls:.2f}")

# 创建一个Excel文件，包含多个工作表
with pd.ExcelWriter('MCM2026 Training Test Problem B/traffic_patterns_analysis.xlsx') as writer:
    # 汇总统计
    all_stats.to_excel(writer, sheet_name='汇总统计', index=False)
    
    # 工作日详细数据
    weekday_patterns = weekday_df['traffic_pattern'].unique()
    for pattern in weekday_patterns:
        pattern_data = weekday_df[weekday_df['traffic_pattern'] == pattern]
        sheet_name = f"工作日_{pattern.replace('/', '_')[:25]}"
        pattern_data.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # 周末详细数据
    weekend_patterns = weekend_df['traffic_pattern'].unique()
    for pattern in weekend_patterns:
        pattern_data = weekend_df[weekend_df['traffic_pattern'] == pattern]
        sheet_name = f"周末_{pattern.replace('/', '_')[:25]}"
        pattern_data.to_excel(writer, sheet_name=sheet_name, index=False)

print("\n已生成以下文件:")
print("1. traffic_patterns_volume_based_stats.csv - 交通模式统计摘要")
print("2. weekday_traffic_patterns_detailed.csv - 工作日详细交通模式数据")
print("3. weekend_traffic_patterns_detailed.csv - 周末详细交通模式数据")
print("4. traffic_patterns_analysis.xlsx - Excel格式的完整分析数据")

# 分析每个交通模式的客流特征
print("\n各交通模式客流特征分析:")

def analyze_pattern_features(df, day_type, pattern_name):
    """分析特定交通模式的客流特征"""
    pattern_data = df[df['traffic_pattern'] == pattern_name]
    if len(pattern_data) == 0:
        return None
    
    # 计算1楼和14楼客流占比
    floor1_ratio = pattern_data['楼层1'].sum() / pattern_data['总次数'].sum() if pattern_data['总次数'].sum() > 0 else 0
    
    floor14_ratio = 0
    if '楼层14' in pattern_data.columns:
        floor14_ratio = pattern_data['楼层14'].sum() / pattern_data['总次数'].sum() if pattern_data['总次数'].sum() > 0 else 0
    
    return {
        'Day_Type': day_type,
        'Pattern': pattern_name,
        'Time_Slots': len(pattern_data),
        'Avg_Total_Calls': pattern_data['总次数'].mean(),
        'Peak_Total_Calls': pattern_data['总次数'].max(),
        'Floor1_Ratio': floor1_ratio,
        'Floor14_Ratio': floor14_ratio,
        'Time_Range': f"{pattern_data['time_of_day'].min()} - {pattern_data['time_of_day'].max()}"
    }

# 分析工作日各模式特征
print("\n工作日各交通模式特征:")
weekday_patterns = weekday_df['traffic_pattern'].unique()
for pattern in weekday_patterns:
    features = analyze_pattern_features(weekday_df, 'Weekday', pattern)
    if features:
        print(f"  {pattern}:")
        print(f"    时间范围: {features['Time_Range']}")
        print(f"    平均客流量: {features['Avg_Total_Calls']:.2f}")
        print(f"    峰值客流量: {features['Peak_Total_Calls']:.2f}")
        print(f"    1楼客流占比: {features['Floor1_Ratio']:.2%}")
        if features['Floor14_Ratio'] > 0:
            print(f"    14楼客流占比: {features['Floor14_Ratio']:.2%}")

print("\n周末各交通模式特征:")
weekend_patterns = weekend_df['traffic_pattern'].unique()
for pattern in weekend_patterns:
    features = analyze_pattern_features(weekend_df, 'Weekend', pattern)
    if features:
        print(f"  {pattern}:")
        print(f"    时间范围: {features['Time_Range']}")
        print(f"    平均客流量: {features['Avg_Total_Calls']:.2f}")
        print(f"    峰值客流量: {features['Peak_Total_Calls']:.2f}")
        print(f"    1楼客流占比: {features['Floor1_Ratio']:.2%}")
        if features['Floor14_Ratio'] > 0:
            print(f"    14楼客流占比: {features['Floor14_Ratio']:.2%}")