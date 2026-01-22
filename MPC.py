"""
电梯MPC停车策略优化系统 v2.0
基于完整的电梯运行数据进行分析和优化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ElevatorDataAnalyzer:
    """电梯数据分析器 - 从原始数据中提取模式"""
    
    def __init__(self):
        self.hall_calls = None
        self.car_calls = None
        self.car_stops = None
        self.car_departures = None
        self.load_changes = None
        self.maintenance = None
        
        # 时间段定义
        self.time_periods = {
            'A': ('07:00', '09:30'),  # 早高峰
            'B': ('09:30', '11:00'),  # 上午
            'C': ('11:00', '13:30'),  # 午餐时间
            'D': ('13:30', '17:00'),  # 下午办公
            'E': ('17:00', '18:00'),  # 晚高峰
            'F': ('18:00', '20:00'),  # 晚餐时间
            'G': ('20:00', '22:00'),  # 晚间
            'H': ('22:00', '07:00')   # 夜间
        }
        
        # 电梯ID列表
        self.elevator_ids = ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I']
        self.num_floors = 14
        self.num_elevators = len(self.elevator_ids)
        
    def _read_csv_with_encoding(self, filepath):
        """尝试多种编码读取CSV文件"""
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                # 其他错误直接抛出
                if 'codec' not in str(e).lower():
                    raise e
                continue
        
        # 最后尝试忽略错误
        return pd.read_csv(filepath, encoding='utf-8', errors='ignore')
    
    def load_data(self):
        """加载所有数据文件"""
        print("正在加载数据...")
        
        # 加载厅外呼叫数据
        self.hall_calls = self._read_csv_with_encoding('MCM2026 Training Test Problem B/hall_calls.csv')
        self.hall_calls['Time'] = pd.to_datetime(self.hall_calls['Time'])
        print(f"  厅外呼叫数据: {len(self.hall_calls)} 条记录")
        
        # 加载轿厢呼叫数据
        self.car_calls = self._read_csv_with_encoding('MCM2026 Training Test Problem B/car_calls.csv')
        self.car_calls['Time'] = pd.to_datetime(self.car_calls['Time'])
        print(f"  轿厢呼叫数据: {len(self.car_calls)} 条记录")
        
        # 加载停靠数据
        self.car_stops = self._read_csv_with_encoding('MCM2026 Training Test Problem B/car_stops.csv')
        self.car_stops['Time'] = pd.to_datetime(self.car_stops['Time'])
        print(f"  停靠数据: {len(self.car_stops)} 条记录")
        
        # 加载出发数据
        self.car_departures = self._read_csv_with_encoding('MCM2026 Training Test Problem B/car_departures.csv')
        self.car_departures['Time'] = pd.to_datetime(self.car_departures['Time'])
        print(f"  出发数据: {len(self.car_departures)} 条记录")
        
        # 加载载重变化数据
        self.load_changes = self._read_csv_with_encoding('MCM2026 Training Test Problem B/load_changes.csv')
        self.load_changes['Time'] = pd.to_datetime(self.load_changes['Time'])
        print(f"  载重变化数据: {len(self.load_changes)} 条记录")
        
        # 加载维护模式数据
        self.maintenance = self._read_csv_with_encoding('MCM2026 Training Test Problem B/maintenance_mode.csv')
        self.maintenance['Time'] = pd.to_datetime(self.maintenance['Time'])
        print(f"  维护模式数据: {len(self.maintenance)} 条记录")
        
        print("数据加载完成!")
        
    def get_time_period(self, time):
        """获取时间对应的时间段"""
        hour = time.hour
        minute = time.minute
        time_minutes = hour * 60 + minute
        
        for period, (start, end) in self.time_periods.items():
            start_h, start_m = map(int, start.split(':'))
            end_h, end_m = map(int, end.split(':'))
            start_min = start_h * 60 + start_m
            end_min = end_h * 60 + end_m
            
            if start_min > end_min:  # 跨天
                if time_minutes >= start_min or time_minutes < end_min:
                    return period
            else:
                if start_min <= time_minutes < end_min:
                    return period
        return 'H'  # 默认夜间
    
    def analyze_hall_call_patterns(self):
        """分析厅外呼叫模式"""
        print("\n分析厅外呼叫模式...")
        
        # 提取有效的楼层呼叫（Floor字段非空）
        valid_calls = self.hall_calls[self.hall_calls['Floor'].notna()].copy()
        
        # 解析楼层（可能是逗号分隔的多个楼层）
        def parse_floors(floor_str):
            if pd.isna(floor_str):
                return []
            try:
                return [int(f.strip()) for f in str(floor_str).split(',') if f.strip().isdigit()]
            except:
                return []
        
        valid_calls['ParsedFloors'] = valid_calls['Floor'].apply(parse_floors)
        
        # 展开楼层列表
        expanded_calls = []
        for _, row in valid_calls.iterrows():
            for floor in row['ParsedFloors']:
                if 1 <= floor <= self.num_floors:
                    expanded_calls.append({
                        'Time': row['Time'],
                        'Elevator ID': row['Elevator ID'],
                        'Direction': row['Direction'],
                        'Floor': floor,
                        'Hour': row['Time'].hour,
                        'DayOfWeek': row['Time'].dayofweek,
                        'Period': self.get_time_period(row['Time'])
                    })
        
        self.expanded_hall_calls = pd.DataFrame(expanded_calls)
        print(f"  展开后的呼叫记录: {len(self.expanded_hall_calls)} 条")
        
        # 按时间段和楼层统计呼叫次数
        self.call_patterns = {}
        for period in self.time_periods.keys():
            period_data = self.expanded_hall_calls[self.expanded_hall_calls['Period'] == period]
            
            # 分工作日和周末
            weekday_data = period_data[period_data['DayOfWeek'] < 5]
            weekend_data = period_data[period_data['DayOfWeek'] >= 5]
            
            # 统计每个楼层的呼叫次数
            weekday_floor_counts = weekday_data.groupby('Floor').size()
            weekend_floor_counts = weekend_data.groupby('Floor').size()
            
            # 填充所有楼层
            weekday_counts = np.zeros(self.num_floors)
            weekend_counts = np.zeros(self.num_floors)
            
            for floor, count in weekday_floor_counts.items():
                if 1 <= floor <= self.num_floors:
                    weekday_counts[floor-1] = count
            for floor, count in weekend_floor_counts.items():
                if 1 <= floor <= self.num_floors:
                    weekend_counts[floor-1] = count
            
            # 计算每5分钟的平均呼叫次数
            period_duration = self._get_period_duration(period)
            num_weekdays = len(weekday_data['Time'].dt.date.unique()) if len(weekday_data) > 0 else 1
            num_weekends = len(weekend_data['Time'].dt.date.unique()) if len(weekend_data) > 0 else 1
            
            weekday_rate = weekday_counts / (num_weekdays * period_duration / 5) if num_weekdays > 0 else weekday_counts
            weekend_rate = weekend_counts / (num_weekends * period_duration / 5) if num_weekends > 0 else weekend_counts
            
            self.call_patterns[period] = {
                'weekday': {
                    'total_calls': weekday_counts.sum(),
                    'floor_counts': weekday_counts,
                    'rate_per_5min': weekday_rate,
                    'distribution': weekday_counts / weekday_counts.sum() if weekday_counts.sum() > 0 else np.ones(self.num_floors) / self.num_floors
                },
                'weekend': {
                    'total_calls': weekend_counts.sum(),
                    'floor_counts': weekend_counts,
                    'rate_per_5min': weekend_rate,
                    'distribution': weekend_counts / weekend_counts.sum() if weekend_counts.sum() > 0 else np.ones(self.num_floors) / self.num_floors
                }
            }
            
            print(f"  {period}段: 工作日{weekday_counts.sum():.0f}次, 周末{weekend_counts.sum():.0f}次")
        
        return self.call_patterns
    
    def _get_period_duration(self, period):
        """获取时间段时长（分钟）"""
        durations = {
            'A': 150, 'B': 90, 'C': 150, 'D': 210,
            'E': 60, 'F': 120, 'G': 120, 'H': 540
        }
        return durations.get(period, 60)
    
    def analyze_elevator_travel_times(self):
        """分析电梯运行时间"""
        print("\n分析电梯运行时间...")
        
        # 合并停靠和出发数据，计算在每层的停留时间
        stops = self.car_stops.copy()
        departures = self.car_departures.copy()
        
        # 按电梯和时间排序
        stops = stops.sort_values(['Elevator ID', 'Time'])
        departures = departures.sort_values(['Elevator ID', 'Time'])
        
        # 计算停留时间
        dwell_times = []
        for elevator in self.elevator_ids:
            elev_stops = stops[stops['Elevator ID'] == elevator]
            elev_deps = departures[departures['Elevator ID'] == elevator]
            
            for _, stop in elev_stops.iterrows():
                # 找到该停靠之后最近的出发
                next_deps = elev_deps[(elev_deps['Time'] > stop['Time']) & 
                                       (elev_deps['Floor'] == stop['Floor'])]
                if len(next_deps) > 0:
                    dep_time = next_deps.iloc[0]['Time']
                    dwell = (dep_time - stop['Time']).total_seconds()
                    if 0 < dwell < 300:  # 过滤异常值（0-5分钟）
                        dwell_times.append({
                            'elevator': elevator,
                            'floor': stop['Floor'],
                            'dwell_time': dwell,
                            'car_call': stop['Stop Reason - Car Call'],
                            'hall_call': stop['Stop Reason - Hall Call']
                        })
        
        self.dwell_times_df = pd.DataFrame(dwell_times)
        
        # 计算平均停留时间
        avg_dwell = self.dwell_times_df['dwell_time'].mean()
        print(f"  平均停留时间: {avg_dwell:.1f}秒")
        
        # 估算层间运行时间（从连续停靠计算）
        travel_times = []
        for elevator in self.elevator_ids:
            elev_stops = stops[stops['Elevator ID'] == elevator].sort_values('Time')
            for i in range(1, len(elev_stops)):
                prev = elev_stops.iloc[i-1]
                curr = elev_stops.iloc[i]
                
                floor_diff = abs(curr['Floor'] - prev['Floor'])
                time_diff = (curr['Time'] - prev['Time']).total_seconds()
                
                if floor_diff > 0 and 0 < time_diff < 120:  # 过滤异常
                    travel_times.append({
                        'floors': floor_diff,
                        'time': time_diff,
                        'speed': floor_diff / time_diff if time_diff > 0 else 0
                    })
        
        self.travel_times_df = pd.DataFrame(travel_times)
        avg_speed = self.travel_times_df['speed'].mean()
        print(f"  估算电梯速度: {avg_speed:.3f}层/秒")
        
        self.elevator_params = {
            'avg_dwell_time': avg_dwell,
            'avg_speed': avg_speed,
            'door_time': 8.0  # 估算开关门时间
        }
        
        return self.elevator_params
    
    def analyze_passenger_load(self):
        """分析乘客载重模式"""
        print("\n分析乘客载重模式...")
        
        loads = self.load_changes.copy()
        loads['Hour'] = loads['Time'].dt.hour
        loads['Period'] = loads['Time'].apply(self.get_time_period)
        loads['DayOfWeek'] = loads['Time'].dt.dayofweek
        
        # 估算人数（假设平均每人70kg）
        avg_weight_per_person = 70
        loads['Passengers_In'] = loads['Load In (kg)'] / avg_weight_per_person
        loads['Passengers_Out'] = loads['Load Out (kg)'] / avg_weight_per_person
        
        # 按时间段和楼层统计
        self.load_patterns = {}
        for period in self.time_periods.keys():
            period_data = loads[loads['Period'] == period]
            
            # 每层的上下客人数
            floor_in = period_data.groupby('Floor')['Passengers_In'].sum()
            floor_out = period_data.groupby('Floor')['Passengers_Out'].sum()
            
            passengers_in = np.zeros(self.num_floors)
            passengers_out = np.zeros(self.num_floors)
            
            for floor, count in floor_in.items():
                if 1 <= floor <= self.num_floors:
                    passengers_in[floor-1] = count
            for floor, count in floor_out.items():
                if 1 <= floor <= self.num_floors:
                    passengers_out[floor-1] = count
            
            self.load_patterns[period] = {
                'passengers_in': passengers_in,
                'passengers_out': passengers_out,
                'net_flow': passengers_in - passengers_out
            }
            
            print(f"  {period}段: 上客{passengers_in.sum():.0f}人, 下客{passengers_out.sum():.0f}人")
        
        return self.load_patterns
    
    def get_peak_floors(self, period, is_weekday=True, top_n=5):
        """获取指定时间段的高需求楼层"""
        day_type = 'weekday' if is_weekday else 'weekend'
        if period in self.call_patterns:
            distribution = self.call_patterns[period][day_type]['distribution']
            top_indices = np.argsort(distribution)[-top_n:][::-1]
            return top_indices + 1  # 转换为1-based楼层号
        return np.array([1, 7, 8, 9, 10])  # 默认值


class ElevatorMPCController:
    """基于MPC的电梯停车控制器 v2.0"""
    
    def __init__(self, data_analyzer):
        """
        初始化MPC控制器
        
        参数:
        - data_analyzer: ElevatorDataAnalyzer实例
        """
        self.analyzer = data_analyzer
        self.E = data_analyzer.num_elevators
        self.F = data_analyzer.num_floors
        self.N = 3  # MPC预测步数
        
        # 从数据分析中获取电梯参数
        if hasattr(data_analyzer, 'elevator_params'):
            self.speed = data_analyzer.elevator_params.get('avg_speed', 0.3)
            self.door_time = data_analyzer.elevator_params.get('door_time', 8.0)
            self.dwell_time = data_analyzer.elevator_params.get('avg_dwell_time', 15.0)
        else:
            self.speed = 0.3
            self.door_time = 8.0
            self.dwell_time = 15.0
        
        # 成本权重
        self.weights = {
            'waiting_time': 10.0,
            'energy': 0.1,
            'movement': 0.5,
            'coverage': 2.0
        }
        
        # 电梯状态
        self.elevator_states = None
        
    def initialize_elevators(self, initial_positions=None):
        """初始化电梯状态"""
        if initial_positions is None:
            # 默认均匀分布
            initial_positions = np.linspace(1, self.F, self.E, dtype=int)
        
        self.elevator_states = []
        for i in range(self.E):
            pos = initial_positions[i] if i < len(initial_positions) else self.F // 2
            self.elevator_states.append({
                'id': self.analyzer.elevator_ids[i] if i < len(self.analyzer.elevator_ids) else f'E{i}',
                'current_floor': int(pos),
                'target_floor': int(pos),
                'direction': 0,
                'status': 'idle',
                'passengers': 0
            })
    
    def predict_demand(self, period, is_weekday=True, num_steps=3):
        """
        基于历史数据预测乘客需求
        
        参数:
        - period: 时间段
        - is_weekday: 是否工作日
        - num_steps: 预测步数
        
        返回:
        - 预测的需求矩阵 (num_steps × F)
        """
        predictions = np.zeros((num_steps, self.F))
        
        day_type = 'weekday' if is_weekday else 'weekend'
        
        if period in self.analyzer.call_patterns:
            base_rate = self.analyzer.call_patterns[period][day_type]['rate_per_5min']
            base_rate = np.maximum(0, np.nan_to_num(base_rate, nan=0.0))
            
            for step in range(num_steps):
                # 添加随机扰动模拟不确定性
                noise = np.random.normal(0, 0.1, self.F)
                predictions[step, :] = np.maximum(0, base_rate + noise)
        
        return predictions
    
    def calculate_expected_wait_time(self, elevator_positions, demand_distribution):
        """
        计算预期等待时间
        
        参数:
        - elevator_positions: 电梯位置数组
        - demand_distribution: 需求分布
        
        返回:
        - 总预期等待时间
        """
        total_wait = 0
        
        for floor in range(self.F):
            demand = demand_distribution[floor]
            if demand > 0:
                # 计算到最近电梯的距离
                distances = [abs(pos - (floor + 1)) for pos in elevator_positions]
                min_distance = min(distances)
                
                # 等待时间 = 移动时间 + 开门时间
                travel_time = min_distance / self.speed if self.speed > 0 else 0
                wait_time = travel_time + self.door_time
                
                total_wait += demand * wait_time
        
        return total_wait
    
    def optimize_parking_positions(self, period, is_weekday=True):
        """
        优化电梯停车位置
        
        参数:
        - period: 时间段
        - is_weekday: 是否工作日
        
        返回:
        - 最优停车位置
        """
        # 获取需求预测
        demand_predictions = self.predict_demand(period, is_weekday, self.N)
        avg_demand = demand_predictions.mean(axis=0)
        
        # 获取当前位置
        current_positions = [e['current_floor'] for e in self.elevator_states]
        
        def objective(positions):
            """目标函数：最小化总成本"""
            positions = np.clip(positions, 1, self.F)
            
            # 等待时间成本
            wait_cost = self.calculate_expected_wait_time(positions, avg_demand)
            
            # 移动成本
            move_cost = sum(abs(positions[i] - current_positions[i]) for i in range(len(positions)))
            
            # 覆盖成本（鼓励电梯分散）
            coverage_cost = 0
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    # 如果两个电梯太近，增加成本
                    dist = abs(positions[i] - positions[j])
                    if dist < 2:
                        coverage_cost += (2 - dist) ** 2
            
            total = (self.weights['waiting_time'] * wait_cost + 
                    self.weights['movement'] * move_cost +
                    self.weights['coverage'] * coverage_cost)
            
            return total
        
        # 初始猜测：基于需求分布
        x0 = self._get_demand_based_initial_positions(avg_demand)
        
        # 约束：楼层范围
        bounds = [(1, self.F) for _ in range(self.E)]
        
        # 优化
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            optimal_positions = np.round(result.x).astype(int)
            optimal_positions = np.clip(optimal_positions, 1, self.F)
        else:
            # 使用启发式方法
            optimal_positions = self._heuristic_parking(avg_demand)
        
        return optimal_positions
    
    def _get_demand_based_initial_positions(self, demand):
        """基于需求分布生成初始位置"""
        positions = []
        
        # 按需求排序楼层
        sorted_floors = np.argsort(demand)[::-1]
        
        for i in range(self.E):
            if i < len(sorted_floors):
                floor = sorted_floors[i % len(sorted_floors)] + 1
            else:
                floor = self.F // 2
            
            # 避免重复
            while floor in positions and len(positions) < self.F:
                floor = floor % self.F + 1
            
            positions.append(floor)
        
        return np.array(positions)
    
    def _heuristic_parking(self, demand):
        """启发式停车策略"""
        positions = []
        
        # 确保demand有效
        demand = np.maximum(0, np.nan_to_num(demand, nan=0.0))
        
        # 按需求排序
        sorted_floors = np.argsort(demand)[::-1]
        
        for i in range(self.E):
            if i < len(sorted_floors):
                floor = int(sorted_floors[i]) + 1
            else:
                floor = self.F // 2
            
            # 避免重复
            while floor in positions:
                floor = (floor % self.F) + 1
            
            positions.append(floor)
        
        return np.array(positions[:self.E])
    
    def update_states(self, new_positions):
        """更新电梯状态"""
        for i, elevator in enumerate(self.elevator_states):
            if i < len(new_positions):
                elevator['target_floor'] = int(new_positions[i])
                elevator['current_floor'] = int(new_positions[i])
                elevator['status'] = 'idle'
    
    def simulate_period(self, period, duration_minutes, is_weekday=True):
        """
        模拟一个时间段的运行
        
        参数:
        - period: 时间段
        - duration_minutes: 时长（分钟）
        - is_weekday: 是否工作日
        
        返回:
        - 性能统计
        """
        total_wait_time = 0
        total_passengers = 0
        total_movement = 0
        decisions = 0
        
        steps = duration_minutes // 5
        
        for step in range(steps):
            # 预测需求
            demand = self.predict_demand(period, is_weekday, 1)[0]
            
            # 优化停车位置
            old_positions = [e['current_floor'] for e in self.elevator_states]
            optimal_positions = self.optimize_parking_positions(period, is_weekday)
            
            # 计算等待时间
            wait = self.calculate_expected_wait_time(optimal_positions, demand)
            total_wait_time += wait
            total_passengers += demand.sum()
            
            # 计算移动距离
            for i in range(len(optimal_positions)):
                if i < len(old_positions):
                    total_movement += abs(optimal_positions[i] - old_positions[i])
            
            # 更新状态
            self.update_states(optimal_positions)
            decisions += 1
        
        avg_wait = total_wait_time / max(total_passengers, 1)
        
        return {
            'avg_waiting_time': avg_wait,
            'total_passengers': total_passengers,
            'total_movement': total_movement,
            'decisions_made': decisions,
            'final_positions': [e['current_floor'] for e in self.elevator_states]
        }


def run_analysis_and_simulation():
    """运行完整的分析和仿真"""
    
    print("=" * 70)
    print("电梯MPC停车策略优化系统 v2.0")
    print("=" * 70)
    
    # 1. 数据分析
    analyzer = ElevatorDataAnalyzer()
    analyzer.load_data()
    
    # 分析数据模式
    call_patterns = analyzer.analyze_hall_call_patterns()
    elevator_params = analyzer.analyze_elevator_travel_times()
    load_patterns = analyzer.analyze_passenger_load()
    
    # 2. 创建MPC控制器
    controller = ElevatorMPCController(analyzer)
    
    # 3. 定义测试策略
    strategies = {
        'MPC优化策略': 'mpc',
        '固定大堂策略': 'lobby',
        '均匀分布策略': 'uniform',
        '需求导向策略': 'demand'
    }
    
    # 时间段时长
    period_durations = {
        'A': 150, 'B': 90, 'C': 150, 'D': 210,
        'E': 60, 'F': 120, 'G': 120, 'H': 540
    }
    
    # 4. 运行仿真
    print("\n" + "=" * 70)
    print("开始仿真测试")
    print("=" * 70)
    
    results = {strategy: {} for strategy in strategies.keys()}
    
    for strategy_name, strategy_type in strategies.items():
        print(f"\n测试策略: {strategy_name}")
        print("-" * 50)
        
        for period in analyzer.time_periods.keys():
            # 设置初始位置
            if strategy_type == 'lobby':
                initial_positions = [1] * controller.E
            elif strategy_type == 'uniform':
                initial_positions = list(np.linspace(2, controller.F-1, controller.E, dtype=int))
            elif strategy_type == 'demand':
                # 基于需求的初始位置
                peak_floors = analyzer.get_peak_floors(period, is_weekday=True, top_n=controller.E)
                initial_positions = list(peak_floors)[:controller.E]
            else:  # MPC
                initial_positions = list(np.linspace(2, controller.F-1, controller.E, dtype=int))
            
            # 确保足够的位置
            while len(initial_positions) < controller.E:
                initial_positions.append(controller.F // 2)
            
            controller.initialize_elevators(initial_positions)
            
            # 运行仿真
            if strategy_type == 'mpc':
                perf = controller.simulate_period(period, period_durations[period], is_weekday=True)
            else:
                # 固定策略仿真
                perf = simulate_fixed_strategy(controller, analyzer, period, 
                                              period_durations[period], strategy_type)
            
            results[strategy_name][period] = perf
            
            print(f"  {period}段: 平均等待{perf['avg_waiting_time']:.1f}秒, "
                  f"乘客{perf['total_passengers']:.0f}人")
    
    # 5. 分析结果
    analyze_results(results, analyzer)
    
    # 6. 生成建议
    generate_recommendations(results, analyzer, controller)
    
    return results, analyzer, controller


def simulate_fixed_strategy(controller, analyzer, period, duration_minutes, strategy_type):
    """模拟固定策略"""
    total_wait = 0
    total_passengers = 0
    total_movement = 0
    
    steps = duration_minutes // 5
    
    # 获取固定位置
    if strategy_type == 'lobby':
        positions = [1] * controller.E
    elif strategy_type == 'uniform':
        positions = list(np.linspace(2, controller.F-1, controller.E, dtype=int))
    elif strategy_type == 'demand':
        peak_floors = analyzer.get_peak_floors(period, top_n=controller.E)
        positions = list(peak_floors)[:controller.E]
    else:
        positions = [controller.F // 2] * controller.E
    
    for step in range(steps):
        # 预测需求
        demand = controller.predict_demand(period, True, 1)[0]
        
        # 计算等待时间
        wait = controller.calculate_expected_wait_time(positions, demand)
        total_wait += wait
        total_passengers += demand.sum()
    
    avg_wait = total_wait / max(total_passengers, 1)
    
    return {
        'avg_waiting_time': avg_wait,
        'total_passengers': total_passengers,
        'total_movement': total_movement,
        'decisions_made': steps,
        'final_positions': positions
    }


def analyze_results(results, analyzer):
    """分析仿真结果"""
    
    print("\n" + "=" * 70)
    print("性能对比分析")
    print("=" * 70)
    
    # 汇总每个策略的性能
    summary = {}
    for strategy_name in results.keys():
        total_weighted_wait = 0
        total_passengers = 0
        
        for period in results[strategy_name]:
            perf = results[strategy_name][period]
            total_weighted_wait += perf['avg_waiting_time'] * perf['total_passengers']
            total_passengers += perf['total_passengers']
        
        avg_wait = total_weighted_wait / max(total_passengers, 1)
        summary[strategy_name] = {
            'avg_waiting': avg_wait,
            'total_passengers': total_passengers
        }
    
    # 打印排名
    print("\n各策略全天平均等待时间排名:")
    sorted_strategies = sorted(summary.items(), key=lambda x: x[1]['avg_waiting'])
    for i, (name, data) in enumerate(sorted_strategies, 1):
        print(f"  {i}. {name}: {data['avg_waiting']:.1f}秒")
    
    # 计算改进
    best_strategy = sorted_strategies[0][0]
    best_wait = sorted_strategies[0][1]['avg_waiting']
    baseline_wait = summary.get('固定大堂策略', {}).get('avg_waiting', best_wait)
    
    if baseline_wait > 0:
        improvement = (baseline_wait - best_wait) / baseline_wait * 100
        print(f"\n最佳策略 '{best_strategy}' 相比固定大堂策略改进: {improvement:.1f}%")
    
    # 可视化
    visualize_results(results, summary, analyzer)


def visualize_results(results, summary, analyzer):
    """可视化结果"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 策略对比
    ax1 = axes[0, 0]
    strategies = list(summary.keys())
    waits = [summary[s]['avg_waiting'] for s in strategies]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6BAA75']
    
    bars = ax1.bar(strategies, waits, color=colors[:len(strategies)])
    ax1.set_ylabel('平均等待时间(秒)')
    ax1.set_title('各策略全天平均等待时间对比')
    ax1.tick_params(axis='x', rotation=15)
    
    for bar, val in zip(bars, waits):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom')
    
    # 2. 各时间段对比
    ax2 = axes[0, 1]
    periods = list(analyzer.time_periods.keys())
    
    for i, strategy in enumerate(strategies[:3]):  # 只显示前3个策略
        period_waits = [results[strategy][p]['avg_waiting_time'] for p in periods]
        ax2.plot(periods, period_waits, 'o-', label=strategy, color=colors[i])
    
    ax2.set_xlabel('时间段')
    ax2.set_ylabel('平均等待时间(秒)')
    ax2.set_title('各时间段等待时间对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 各楼层呼叫分布（工作日A段为例）
    ax3 = axes[1, 0]
    if 'A' in analyzer.call_patterns:
        dist = analyzer.call_patterns['A']['weekday']['distribution']
        floors = np.arange(1, analyzer.num_floors + 1)
        ax3.bar(floors, dist, color='#2E86AB', alpha=0.7)
        ax3.set_xlabel('楼层')
        ax3.set_ylabel('呼叫占比')
        ax3.set_title('早高峰(A段)各楼层呼叫分布')
        ax3.set_xticks(floors)
    
    # 4. 乘客流量分析
    ax4 = axes[1, 1]
    if hasattr(analyzer, 'load_patterns'):
        periods_list = list(analyzer.load_patterns.keys())
        passengers_in = [analyzer.load_patterns[p]['passengers_in'].sum() for p in periods_list]
        passengers_out = [analyzer.load_patterns[p]['passengers_out'].sum() for p in periods_list]
        
        x = np.arange(len(periods_list))
        width = 0.35
        
        ax4.bar(x - width/2, passengers_in, width, label='上客', color='#2E86AB')
        ax4.bar(x + width/2, passengers_out, width, label='下客', color='#A23B72')
        
        ax4.set_xlabel('时间段')
        ax4.set_ylabel('乘客数量')
        ax4.set_title('各时间段乘客流量')
        ax4.set_xticks(x)
        ax4.set_xticklabels(periods_list)
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('elevator_mpc_analysis.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存为 'elevator_mpc_analysis.png'")
    plt.show()


def generate_recommendations(results, analyzer, controller):
    """生成停车策略建议"""
    
    print("\n" + "=" * 70)
    print("停车策略建议")
    print("=" * 70)
    
    mpc_results = results.get('MPC优化策略', {})
    
    for period, (start, end) in analyzer.time_periods.items():
        print(f"\n{period}段 ({start}-{end}):")
        
        # 获取高需求楼层
        if period in analyzer.call_patterns:
            dist = analyzer.call_patterns[period]['weekday']['distribution']
            top_floors = np.argsort(dist)[-5:][::-1] + 1
            print(f"  高需求楼层: {list(top_floors)}")
        
        # MPC建议位置
        if period in mpc_results:
            positions = mpc_results[period]['final_positions']
            print(f"  MPC建议停车位置: {sorted(positions)}")
            print(f"  预期等待时间: {mpc_results[period]['avg_waiting_time']:.1f}秒")
        
        # 特定时段建议
        if period in ['A', 'E']:
            print("  策略建议: 高峰时段，增加大堂和主要办公楼层覆盖")
        elif period in ['C', 'F']:
            print("  策略建议: 餐饮时段，覆盖餐饮楼层和大堂")
        elif period == 'H':
            print("  策略建议: 夜间低需求，集中在中间楼层节能")
        else:
            print("  策略建议: 常规时段，均匀分布响应随机需求")


# 主程序
if __name__ == "__main__":
    results, analyzer, controller = run_analysis_and_simulation()
