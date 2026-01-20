import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

# 使中文可用
all_fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = []
chinese_font_names = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'SimSun',
                      'NSimSun', 'YouYuan', 'STXihei', 'STKaiti', 'STSong']
for font in chinese_font_names:
    if any(font in f for f in all_fonts):
        chinese_fonts.append(font)

if chinese_fonts:
    print(f"Available Chinese fonts: {chinese_fonts}")
    plt.rcParams['font.sans-serif'] = [chinese_fonts[0]]
else:
    print("No Chinese fonts found, using English only")
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件 - 使用原始字符串
file_path = r"C:\Users\lvsiy\Desktop\模拟MCM\MCM2026 Training Test Problem B\mcm26Train-B-Data\hall_calls.csv"
df = pd.read_csv(file_path)

# 将时间列转换为datetime格式
df['Time'] = pd.to_datetime(df['Time'])

# 设置起始时间为数据中的最早时间，向下取整到5分钟
start_time = df['Time'].min().floor('5min')

# 创建时间区间，每5分钟一个区间
time_bins = pd.date_range(start=start_time, end=df['Time'].max(), freq='5min')

# 统计每个时间区间的呼叫次数
df['Time_Bin'] = pd.cut(df['Time'], bins=time_bins, right=True, include_lowest=True)
calls_per_5min = df.groupby('Time_Bin').size().reset_index(name='Call_Count')
calls_per_5min['Call_Count'] = calls_per_5min['Call_Count'] / 2

# 转换时间区间为更易读的格式
calls_per_5min['Time_Range'] = calls_per_5min['Time_Bin'].apply(
    lambda x: f"{x.left.strftime('%Y-%m-%d %H:%M')} - {x.right.strftime('%H:%M')}"
)

# 输出结果
print("每5分钟呼叫次数统计（前20行）：")
print("=" * 60)
for idx, row in calls_per_5min.head(20).iterrows():
    print(f"{row['Time_Range']:35} | {row['Call_Count']:3} 次")

# 统计汇总信息
print("\n统计摘要：")
print("=" * 60)
print(f"总时间区间数: {len(calls_per_5min)}")
print(f"总呼叫次数: {calls_per_5min['Call_Count'].sum()}")
print(f"平均每5分钟呼叫次数: {calls_per_5min['Call_Count'].mean():.2f}")
print(f"最大5分钟呼叫次数: {calls_per_5min['Call_Count'].max()}")
print(f"最小5分钟呼叫次数: {calls_per_5min['Call_Count'].min()}")

# 确定保存路径
# 保存到桌面
output_path = os.path.join(os.path.expanduser("~"), "Desktop", "calls_per_5min.csv")
# 或者保存到原始文件所在目录
output_path_original = os.path.join(os.path.dirname(file_path), "calls_per_5min.csv")

# 保存到桌面
calls_per_5min[['Time_Range', 'Call_Count']].to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到桌面: {output_path}")

# 同时保存到原始文件所在目录
calls_per_5min[['Time_Range', 'Call_Count']].to_csv(output_path_original, index=False, encoding='utf-8-sig')
print(f"同时保存到原始文件目录: {output_path_original}")

# 显示文件大小信息
import os
if os.path.exists(output_path):
    file_size = os.path.getsize(output_path)
    print(f"生成的文件大小: {file_size} 字节")

# 显示前5行数据预览
print("\n生成文件的前5行数据预览：")
print(calls_per_5min[['Time_Range', 'Call_Count']].head())

# 可视化（可选）
plt.figure(figsize=(15, 6))
plt.bar(range(len(calls_per_5min)), calls_per_5min['Call_Count'])
plt.xlabel('时间区间索引（每5分钟）')
plt.ylabel('呼叫次数')
plt.title('每5分钟大厅呼叫次数分布')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存图表
chart_path = os.path.join(os.path.expanduser("~"), "Desktop", "calls_per_5min_chart.png")
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
print(f"\n图表已保存到桌面: {chart_path}")

#plt.show()
