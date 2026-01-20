# 简化版 - 直接运行
import pandas as pd
import os

# 设置文件路径
file_path = "MCM2026 Training Test Problem B/hall_calls.csv"

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"错误: 找不到文件 '{file_path}'")
    print("请确保文件路径正确，或者将文件放在当前目录下")
    exit()

print(f"正在处理文件: {file_path}")

# 读取数据
df = pd.read_csv(file_path)

# 清理数据
df_clean = df.dropna(subset=['Floor'])
if 'Floor' in df_clean.columns:
    df_clean['Floor'] = pd.to_numeric(df_clean['Floor'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Floor'])
    df_clean['Floor'] = df_clean['Floor'].astype(int)

# 解析时间
df_clean['Time'] = pd.to_datetime(df_clean['Time'], errors='coerce')
df_clean = df_clean.dropna(subset=['Time'])

# 创建5分钟时间槽
df_clean['Time_Slot'] = df_clean['Time'].dt.floor('5min')

# 统计每5分钟的总数
time_slot_counts = df_clean.groupby('Time_Slot').size().reset_index(name='Total_Calls')

# 统计楼层分布
floor_distribution = df_clean.groupby(['Time_Slot', 'Floor']).size().reset_index(name='Count')

# 创建透视表
pivot_table = floor_distribution.pivot_table(
    index='Time_Slot',
    columns='Floor',
    values='Count',
    fill_value=0
)

# 合并结果
result_df = pd.concat([time_slot_counts.set_index('Time_Slot'), pivot_table], axis=1)
result_df = result_df.reset_index()

# 重命名列
column_names = ['时间段'] + ['总次数']
if not pivot_table.columns.empty:
    column_names += [f'楼层{int(col)}' for col in pivot_table.columns]
result_df.columns = column_names

# 保存结果
output_dir = os.path.dirname(file_path)
if not output_dir:
    output_dir = '.'

summary_path = os.path.join(output_dir, 'hall_calls_5min_summary.csv')
result_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

floor_dist_path = os.path.join(output_dir, 'hall_calls_floor_distribution.csv')
floor_distribution.to_csv(floor_dist_path, index=False, encoding='utf-8-sig')

print(f"\n处理完成!")
print(f"原始数据行数: {len(df)}")
print(f"有效数据行数: {len(df_clean)}")
print(f"时间段数: {len(time_slot_counts)}")
print(f"涉及楼层数: {df_clean['Floor'].nunique()}")

print(f"\n输出文件:")
print(f"1. {summary_path}")
print(f"2. {floor_dist_path}")

print(f"\n前5个时间段:")
print(result_df.head().to_string(index=False))