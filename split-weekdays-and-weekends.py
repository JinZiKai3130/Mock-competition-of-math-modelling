import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# 读取数据
df = pd.read_csv('MCM2026 Training Test Problem B/hall_calls_5min_summary.csv')

# 确保时间段列是datetime类型
df['时间段'] = pd.to_datetime(df['时间段'])

# 根据2025-11-03是周一，推算其他日期是周几
# 创建函数判断工作日/周末
def is_weekend(date):
    # 已知2025-11-03是周一 (weekday=0)
    # 计算从2025-11-03到给定日期的天数差，然后推算星期几
    reference_date = datetime(2025, 11, 3)  # 周一
    delta_days = (date.date() - reference_date.date()).days
    
    # 计算星期几 (0=周一, 1=周二, ..., 6=周日)
    weekday_num = (0 + delta_days) % 7  # 从周一开始计算
    
    # 返回是否是周末 (周六=5, 周日=6)
    return weekday_num >= 5

# 添加工作日/周末标志列
df['is_weekend'] = df['时间段'].apply(is_weekend)

# 创建一天中的时间列 (只保留时分，忽略日期)
df['time_of_day'] = df['时间段'].dt.strftime('%H:%M')

# 分别处理工作日和周末的数据
# 工作日数据
weekday_df = df[~df['is_weekend']]
# 周末数据
weekend_df = df[df['is_weekend']]

# 按一天中的时间分组计算平均值
# 工作日平均值
weekday_avg = weekday_df.groupby('time_of_day').agg({
    '总次数': 'mean',
    '楼层1': 'mean',
    '楼层2': 'mean',
    '楼层3': 'mean',
    '楼层4': 'mean',
    '楼层5': 'mean',
    '楼层6': 'mean',
    '楼层7': 'mean',
    '楼层8': 'mean',
    '楼层9': 'mean',
    '楼层10': 'mean',
    '楼层11': 'mean',
    '楼层12': 'mean',
    '楼层13': 'mean',
    '楼层14': 'mean'
}).round(2)  # 保留两位小数

# 周末平均值
weekend_avg = weekend_df.groupby('time_of_day').agg({
    '总次数': 'mean',
    '楼层1': 'mean',
    '楼层2': 'mean',
    '楼层3': 'mean',
    '楼层4': 'mean',
    '楼层5': 'mean',
    '楼层6': 'mean',
    '楼层7': 'mean',
    '楼层8': 'mean',
    '楼层9': 'mean',
    '楼层10': 'mean',
    '楼层11': 'mean',
    '楼层12': 'mean',
    '楼层13': 'mean',
    '楼层14': 'mean'
}).round(2)  # 保留两位小数

# 重置索引，使time_of_day成为列
weekday_avg = weekday_avg.reset_index()
weekend_avg = weekend_avg.reset_index()

# 确保时间列按时间顺序排序
# 先将时间列转换为datetime.time格式用于排序
weekday_avg['time_sort'] = pd.to_datetime(weekday_avg['time_of_day'], format='%H:%M').dt.time
weekday_avg = weekday_avg.sort_values('time_sort')
weekday_avg = weekday_avg.drop('time_sort', axis=1)

weekend_avg['time_sort'] = pd.to_datetime(weekend_avg['time_of_day'], format='%H:%M').dt.time
weekend_avg = weekend_avg.sort_values('time_sort')
weekend_avg = weekend_avg.drop('time_sort', axis=1)

# 保存到两个不同的CSV文件
weekday_avg.to_csv('MCM2026 Training Test Problem B/hall_calls_5min_weekday_avg.csv', index=False, encoding='utf-8-sig')
weekend_avg.to_csv('MCM2026 Training Test Problem B/hall_calls_5min_weekend_avg.csv', index=False, encoding='utf-8-sig')

print("数据已成功处理并保存到两个文件:")
print(f"  工作日平均值: hall_calls_5min_weekday_avg.csv")
print(f"  周末平均值: hall_calls_5min_weekend_avg.csv")
print(f"工作日数据条数: {len(weekday_df)}")
print(f"周末数据条数: {len(weekend_df)}")
print(f"工作日时间点数: {len(weekday_avg)}")
print(f"周末时间点数: {len(weekend_avg)}")

# 显示前几行结果预览
print("\n工作日平均值前10行预览:")
print(weekday_avg.head(10))
print("\n周末平均值前10行预览:")
print(weekend_avg.head(10))