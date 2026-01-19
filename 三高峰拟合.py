import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import matplotlib.font_manager as fm

# 获取所有可用字体
all_fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = []

# 常见中文字体名称
chinese_font_names = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'SimSun',
                      'NSimSun', 'YouYuan', 'STXihei', 'STKaiti', 'STSong']

# 检查系统中有哪些中文字体
for font in chinese_font_names:
    if any(font in f for f in all_fonts):
        chinese_fonts.append(font)

if chinese_fonts:
    print(f"Available Chinese fonts: {chinese_fonts}")
    # 使用第一个找到的中文字体
    plt.rcParams['font.sans-serif'] = [chinese_fonts[0]]
else:
    print("No Chinese fonts found, using English only")
    # 如果没有中文字体，使用英文

plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
# 请确保路径正确，或根据实际文件位置修改
file_path = r"C:\Users\lvsiy\Desktop\calls_per_5min.csv"
df = pd.read_csv(file_path)

# 2. 预处理：提取时间序号 x (1-288) 和 日期
# 修正点：将 .dt.min 改为 .dt.minute
df['Timestamp'] = pd.to_datetime(df['Time_Range'].str.split(' - ').str[0])
df['Date'] = df['Timestamp'].dt.date
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek  # 0=周一, 4=周五
df['x'] = (df['Timestamp'].dt.hour * 60 + df['Timestamp'].dt.minute) // 5 + 1

# 3. 筛选前三周的工作日 (假设数据始于11月初，此处日期请根据实际数据范围微调)
# 根据之前的分析，我们选择 11月3日（周一）到 11月21日（周五）
start_date = pd.to_datetime("2025-11-03").date()
end_date = pd.to_datetime("2025-11-21").date()
workday_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['DayOfWeek'] < 5)]

# 计算所有选定工作日在每个 x 阶段的平均呼叫数
daily_avg = workday_df.groupby('x')['Call_Count'].mean().reset_index()
x_data = daily_avg['x'].values
y_data = daily_avg['Call_Count'].values


# 4. 定义三高峰（高斯混合）模型函数
def triple_peak_model(x, b, a1, mu1, sig1, a2, mu2, sig2, a3, mu3, sig3):
    p1 = a1 * np.exp(-(x - mu1) ** 2 / (2 * sig1 ** 2))
    p2 = a2 * np.exp(-(x - mu2) ** 2 / (2 * sig2 ** 2))
    p3 = a3 * np.exp(-(x - mu3) ** 2 / (2 * sig3 ** 2))
    return b + p1 + p2 + p3


# 5. 执行拟合
# 初始猜想值 [底噪, 幅度1, 位置1, 宽度1, 幅度2, 位置2, 宽度2, 幅度3, 位置3, 宽度3]
# 位置参考：9:30->114, 15:30->186, 20:00->240
initial_guess = [2.5, 18, 114, 15, 26, 186, 25, 12, 240, 12]

try:
    popt, pcov = curve_fit(triple_peak_model, x_data, y_data, p0=initial_guess)

    # 6. 输出结果
    labels = ['底噪b', '幅度A1', '位置μ1', '宽度σ1', '幅度A2', '位置μ2', '宽度σ2', '幅度A3', '位置μ3', '宽度σ3']
    print("拟合系数结果：")
    for label, value in zip(labels, popt):
        print(f"{label}: {value:.4f}")

    # 7. 可视化
    plt.figure(figsize=(12, 6))
    plt.scatter(x_data, y_data, s=10, color='gray', alpha=0.5, label='实际均值数据')
    plt.plot(x_data, triple_peak_model(x_data, *popt), 'r-', linewidth=2, label='三高峰拟合曲线')
    plt.title("Workday Call Volume Triple-Peak Fitting")
    plt.xlabel("Time interval (x=1 to 288)")
    plt.ylabel("Call Count")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

except Exception as e:
    print(f"拟合过程中出现错误: {e}")