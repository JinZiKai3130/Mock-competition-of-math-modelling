"""
电梯MPC停车策略优化系统 v3.0
修复训练/测试数据分离问题
- 训练集：用于学习需求模式
- 测试集：用于评估策略性能（使用真实呼叫数据）
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
    """电梯数据分析器 - 支持训练/测试分割"""
    
    def __init__(self, train_ratio=0.7):
        """
        初始化分析器
        
        参数:
        - train_ratio: 训练集比例 (默认70%)
        """
        self.train_ratio = train_ratio
        
        # 原始数据
        self.hall_calls = None
        self.car_calls = None
        self.car_stops = None
        self.car_departures = None
        self.load_changes = None
        self.maintenance = None
        
        # 分割后的数据
        self.train_hall_calls = None
        self.test_hall_calls = None
        self.train_dates = None
        self.test_dates = None
        
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
                if 'codec' not in str(e).lower():
                    raise e
                continue
        
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
        
    def split_train_test(self):
        """
        按日期分割训练集和测试集
        
        关键：确保模型只用训练数据学习，测试数据用于验证
        """
        print(f"\n按日期分割数据 (训练集{self.train_ratio*100:.0f}% / 测试集{(1-self.train_ratio)*100:.0f}%)...")
        
        # 获取所有唯一日期
        all_dates = sorted(self.hall_calls['Time'].dt.date.unique())
        n_dates = len(all_dates)
        
        # 分割日期
        split_idx = int(n_dates * self.train_ratio)
        self.train_dates = set(all_dates[:split_idx])
        self.test_dates = set(all_dates[split_idx:])
        
        print(f"  总天数: {n_dates}")
        print(f"  训练集: {len(self.train_dates)} 天 ({min(self.train_dates)} 至 {max(self.train_dates)})")
        print(f"  测试集: {len(self.test_dates)} 天 ({min(self.test_dates)} 至 {max(self.test_dates)})")
        
        # 分割厅外呼叫数据
        self.hall_calls['Date'] = self.hall_calls['Time'].dt.date
        self.train_hall_calls = self.hall_calls[self.hall_calls['Date'].isin(self.train_dates)].copy()
        self.test_hall_calls = self.hall_calls[self.hall_calls['Date'].isin(self.test_dates)].copy()
        
        print(f"  训练集呼叫: {len(self.train_hall_calls)} 条")
        print(f"  测试集呼叫: {len(self.test_hall_calls)} 条")
        
        # 分割停靠数据（用于评估）
        self.car_stops['Date'] = self.car_stops['Time'].dt.date
        self.train_car_stops = self.car_stops[self.car_stops['Date'].isin(self.train_dates)].copy()
        self.test_car_stops = self.car_stops[self.car_stops['Date'].isin(self.test_dates)].copy()
        
        # 分割出发数据
        self.car_departures['Date'] = self.car_departures['Time'].dt.date
        self.train_car_departures = self.car_departures[self.car_departures['Date'].isin(self.train_dates)].copy()
        self.test_car_departures = self.car_departures[self.car_departures['Date'].isin(self.test_dates)].copy()
        
        return self.train_dates, self.test_dates
        
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
        return 'H'
    
    def analyze_hall_call_patterns(self, use_train_only=True):
        """
        分析厅外呼叫模式
        
        参数:
        - use_train_only: 是否只使用训练集数据（默认True）
        """
        data_source = "训练集" if use_train_only else "全部数据"
        print(f"\n分析厅外呼叫模式 (使用{data_source})...")
        
        # 选择数据源
        if use_train_only and self.train_hall_calls is not None:
            hall_calls_data = self.train_hall_calls
        else:
            hall_calls_data = self.hall_calls
        
        # 提取有效的楼层呼叫
        valid_calls = hall_calls_data[hall_calls_data['Floor'].notna()].copy()
        
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
        
        # 按时间段和楼层统计
        self.call_patterns = {}
        for period in self.time_periods.keys():
            period_data = self.expanded_hall_calls[self.expanded_hall_calls['Period'] == period]
            
            weekday_data = period_data[period_data['DayOfWeek'] < 5]
            weekend_data = period_data[period_data['DayOfWeek'] >= 5]
            
            weekday_floor_counts = weekday_data.groupby('Floor').size()
            weekend_floor_counts = weekend_data.groupby('Floor').size()
            
            weekday_counts = np.zeros(self.num_floors)
            weekend_counts = np.zeros(self.num_floors)
            
            for floor, count in weekday_floor_counts.items():
                if 1 <= floor <= self.num_floors:
                    weekday_counts[floor-1] = count
            for floor, count in weekend_floor_counts.items():
                if 1 <= floor <= self.num_floors:
                    weekend_counts[floor-1] = count
            
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
    
    def analyze_elevator_travel_times(self, use_train_only=True):
        """分析电梯运行时间（仅使用训练集）"""
        print(f"\n分析电梯运行时间 ({'训练集' if use_train_only else '全部数据'})...")
        
        if use_train_only and hasattr(self, 'train_car_stops'):
            stops = self.train_car_stops.copy()
            departures = self.train_car_departures.copy()
        else:
            stops = self.car_stops.copy()
            departures = self.car_departures.copy()
        
        stops = stops.sort_values(['Elevator ID', 'Time'])
        departures = departures.sort_values(['Elevator ID', 'Time'])
        
        # 计算停留时间
        dwell_times = []
        for elevator in self.elevator_ids:
            elev_stops = stops[stops['Elevator ID'] == elevator]
            elev_deps = departures[departures['Elevator ID'] == elevator]
            
            for _, stop in elev_stops.iterrows():
                next_deps = elev_deps[(elev_deps['Time'] > stop['Time']) & 
                                       (elev_deps['Floor'] == stop['Floor'])]
                if len(next_deps) > 0:
                    dep_time = next_deps.iloc[0]['Time']
                    dwell = (dep_time - stop['Time']).total_seconds()
                    if 0 < dwell < 300:
                        dwell_times.append({
                            'elevator': elevator,
                            'floor': stop['Floor'],
                            'dwell_time': dwell
                        })
        
        self.dwell_times_df = pd.DataFrame(dwell_times)
        avg_dwell = self.dwell_times_df['dwell_time'].mean() if len(dwell_times) > 0 else 15.0
        print(f"  平均停留时间: {avg_dwell:.1f}秒")
        
        # 估算层间运行时间
        travel_times = []
        for elevator in self.elevator_ids:
            elev_stops = stops[stops['Elevator ID'] == elevator].sort_values('Time')
            for i in range(1, len(elev_stops)):
                prev = elev_stops.iloc[i-1]
                curr = elev_stops.iloc[i]
                
                floor_diff = abs(curr['Floor'] - prev['Floor'])
                time_diff = (curr['Time'] - prev['Time']).total_seconds()
                
                if floor_diff > 0 and 0 < time_diff < 120:
                    travel_times.append({
                        'floors': floor_diff,
                        'time': time_diff,
                        'speed': floor_diff / time_diff if time_diff > 0 else 0
                    })
        
        self.travel_times_df = pd.DataFrame(travel_times)
        avg_speed = self.travel_times_df['speed'].mean() if len(travel_times) > 0 else 0.3
        print(f"  估算电梯速度: {avg_speed:.3f}层/秒")
        
        self.elevator_params = {
            'avg_dwell_time': avg_dwell,
            'avg_speed': avg_speed,
            'door_time': 8.0
        }
        
        return self.elevator_params
    
    def get_peak_floors(self, period, is_weekday=True, top_n=5):
        """获取指定时间段的高需求楼层（基于训练集模式）"""
        day_type = 'weekday' if is_weekday else 'weekend'
        if period in self.call_patterns:
            distribution = self.call_patterns[period][day_type]['distribution']
            top_indices = np.argsort(distribution)[-top_n:][::-1]
            return top_indices + 1
        return np.array([1, 7, 8, 9, 10])
    
    def get_test_calls_for_period(self, period, date, is_weekday=True):
        """
        获取测试集中指定时段的真实呼叫数据
        
        参数:
        - period: 时间段
        - date: 日期
        - is_weekday: 是否工作日
        
        返回:
        - DataFrame: 该时段的真实呼叫记录
        """
        if self.test_hall_calls is None:
            return pd.DataFrame()
        
        # 筛选指定日期
        day_calls = self.test_hall_calls[self.test_hall_calls['Date'] == date].copy()
        
        if len(day_calls) == 0:
            return pd.DataFrame()
        
        # 添加时间段
        day_calls['Period'] = day_calls['Time'].apply(self.get_time_period)
        
        # 筛选指定时段
        period_calls = day_calls[day_calls['Period'] == period].copy()
        
        return period_calls


class ElevatorMPCController:
    """基于MPC的电梯停车控制器 v3.0"""
    
    def __init__(self, data_analyzer):
        self.analyzer = data_analyzer
        self.E = data_analyzer.num_elevators
        self.F = data_analyzer.num_floors
        self.N = 3
        
        if hasattr(data_analyzer, 'elevator_params'):
            self.speed = data_analyzer.elevator_params.get('avg_speed', 0.3)
            self.door_time = data_analyzer.elevator_params.get('door_time', 8.0)
            self.dwell_time = data_analyzer.elevator_params.get('avg_dwell_time', 15.0)
        else:
            self.speed = 0.3
            self.door_time = 8.0
            self.dwell_time = 15.0
        
        self.weights = {
            'waiting_time': 10.0,
            'energy': 0.1,
            'movement': 0.5,
            'coverage': 2.0
        }
        
        self.elevator_states = None
        
    def initialize_elevators(self, initial_positions=None):
        """初始化电梯状态"""
        if initial_positions is None:
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
        """基于训练集模式预测需求"""
        predictions = np.zeros((num_steps, self.F))
        
        day_type = 'weekday' if is_weekday else 'weekend'
        
        if period in self.analyzer.call_patterns:
            base_rate = self.analyzer.call_patterns[period][day_type]['rate_per_5min']
            base_rate = np.maximum(0, np.nan_to_num(base_rate, nan=0.0))
            
            for step in range(num_steps):
                noise = np.random.normal(0, 0.1, self.F)
                predictions[step, :] = np.maximum(0, base_rate + noise)
        
        return predictions
    
    def calculate_expected_wait_time(self, elevator_positions, demand_distribution):
        """计算预期等待时间"""
        total_wait = 0
        
        for floor in range(self.F):
            demand = demand_distribution[floor]
            if demand > 0:
                distances = [abs(pos - (floor + 1)) for pos in elevator_positions]
                min_distance = min(distances)
                travel_time = min_distance / self.speed if self.speed > 0 else 0
                wait_time = travel_time + self.door_time
                total_wait += demand * wait_time
        
        return total_wait
    
    def calculate_actual_wait_time(self, elevator_positions, call_floors):
        """
        计算真实呼叫的等待时间
        
        参数:
        - elevator_positions: 电梯当前位置
        - call_floors: 真实呼叫楼层列表
        
        返回:
        - 总等待时间, 呼叫次数
        """
        total_wait = 0
        num_calls = len(call_floors)
        
        for floor in call_floors:
            if 1 <= floor <= self.F:
                distances = [abs(pos - floor) for pos in elevator_positions]
                min_distance = min(distances)
                travel_time = min_distance / self.speed if self.speed > 0 else 0
                wait_time = travel_time + self.door_time
                total_wait += wait_time
        
        return total_wait, num_calls
    
    def optimize_parking_positions(self, period, is_weekday=True):
        """优化电梯停车位置（基于训练集学习的模式）"""
        demand_predictions = self.predict_demand(period, is_weekday, self.N)
        avg_demand = demand_predictions.mean(axis=0)
        
        current_positions = [e['current_floor'] for e in self.elevator_states]
        
        def objective(positions):
            positions = np.clip(positions, 1, self.F)
            
            wait_cost = self.calculate_expected_wait_time(positions, avg_demand)
            move_cost = sum(abs(positions[i] - current_positions[i]) for i in range(len(positions)))
            
            coverage_cost = 0
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    dist = abs(positions[i] - positions[j])
                    if dist < 2:
                        coverage_cost += (2 - dist) ** 2
            
            total = (self.weights['waiting_time'] * wait_cost + 
                    self.weights['movement'] * move_cost +
                    self.weights['coverage'] * coverage_cost)
            
            return total
        
        x0 = self._get_demand_based_initial_positions(avg_demand)
        bounds = [(1, self.F) for _ in range(self.E)]
        
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            optimal_positions = np.round(result.x).astype(int)
            optimal_positions = np.clip(optimal_positions, 1, self.F)
        else:
            optimal_positions = self._heuristic_parking(avg_demand)
        
        return optimal_positions
    
    def _get_demand_based_initial_positions(self, demand):
        """基于需求分布生成初始位置"""
        positions = []
        sorted_floors = np.argsort(demand)[::-1]
        
        for i in range(self.E):
            if i < len(sorted_floors):
                floor = sorted_floors[i % len(sorted_floors)] + 1
            else:
                floor = self.F // 2
            
            while floor in positions and len(positions) < self.F:
                floor = floor % self.F + 1
            
            positions.append(floor)
        
        return np.array(positions)
    
    def _heuristic_parking(self, demand):
        """启发式停车策略"""
        positions = []
        demand = np.maximum(0, np.nan_to_num(demand, nan=0.0))
        sorted_floors = np.argsort(demand)[::-1]
        
        for i in range(self.E):
            if i < len(sorted_floors):
                floor = int(sorted_floors[i]) + 1
            else:
                floor = self.F // 2
            
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


def parse_call_floors(hall_calls_df, num_floors=14):
    """解析呼叫数据中的楼层"""
    floors = []
    
    for _, row in hall_calls_df.iterrows():
        floor_str = row.get('Floor', '')
        if pd.isna(floor_str):
            continue
        try:
            for f in str(floor_str).split(','):
                f = f.strip()
                if f.isdigit():
                    floor = int(f)
                    if 1 <= floor <= num_floors:
                        floors.append(floor)
        except:
            continue
    
    return floors


def evaluate_strategy_on_test_set(controller, analyzer, strategy_type, strategy_name):
    """
    在测试集上评估策略
    
    关键改进：使用测试集的真实呼叫数据评估，而非预测数据
    
    参数:
    - controller: MPC控制器
    - analyzer: 数据分析器
    - strategy_type: 策略类型
    - strategy_name: 策略名称
    
    返回:
    - 各时段的性能结果
    """
    print(f"\n在测试集上评估: {strategy_name}")
    print("-" * 50)
    
    results = {}
    
    # 遍历测试集的每一天
    test_dates = sorted(analyzer.test_dates)
    
    for period in analyzer.time_periods.keys():
        total_wait_time = 0
        total_calls = 0
        total_movement = 0
        
        # 设置该策略的电梯位置
        if strategy_type == 'lobby':
            positions = [1] * controller.E
        elif strategy_type == 'uniform':
            positions = list(np.linspace(2, controller.F-1, controller.E, dtype=int))
        elif strategy_type == 'demand':
            peak_floors = analyzer.get_peak_floors(period, is_weekday=True, top_n=controller.E)
            positions = list(peak_floors)[:controller.E]
            while len(positions) < controller.E:
                positions.append(controller.F // 2)
        elif strategy_type == 'mpc':
            # MPC策略：使用训练集模式优化
            controller.initialize_elevators(list(np.linspace(2, controller.F-1, controller.E, dtype=int)))
            positions = list(controller.optimize_parking_positions(period, is_weekday=True))
        else:
            positions = [controller.F // 2] * controller.E
        
        # 确保位置数量正确
        while len(positions) < controller.E:
            positions.append(controller.F // 2)
        
        # 在测试集每一天评估
        for date in test_dates:
            is_weekday = date.weekday() < 5
            
            # 获取该日期该时段的真实呼叫
            period_calls = analyzer.get_test_calls_for_period(period, date, is_weekday)
            
            if len(period_calls) > 0:
                # 解析真实呼叫楼层
                call_floors = parse_call_floors(period_calls, controller.F)
                
                # 计算真实等待时间
                wait, num = controller.calculate_actual_wait_time(positions, call_floors)
                total_wait_time += wait
                total_calls += num
        
        # 计算平均等待时间
        avg_wait = total_wait_time / max(total_calls, 1)
        
        results[period] = {
            'avg_waiting_time': avg_wait,
            'total_calls': total_calls,
            'total_movement': total_movement,
            'final_positions': positions
        }
        
        print(f"  {period}段: 平均等待{avg_wait:.1f}秒, 呼叫{total_calls}次")
    
    return results


def run_analysis_and_simulation():
    """运行完整的分析和仿真（带训练/测试分割）"""
    
    print("=" * 70)
    print("电梯MPC停车策略优化系统 v3.0")
    print("训练/测试数据分离版本")
    print("=" * 70)
    
    # 1. 数据分析（带训练/测试分割）
    analyzer = ElevatorDataAnalyzer(train_ratio=0.7)  # 70%训练，30%测试
    analyzer.load_data()
    analyzer.split_train_test()
    
    # 2. 仅用训练集分析模式
    print("\n" + "=" * 70)
    print("步骤1: 使用训练集学习需求模式")
    print("=" * 70)
    call_patterns = analyzer.analyze_hall_call_patterns(use_train_only=True)
    elevator_params = analyzer.analyze_elevator_travel_times(use_train_only=True)
    
    # 3. 创建MPC控制器
    controller = ElevatorMPCController(analyzer)
    
    # 4. 定义测试策略
    strategies = {
        'MPC优化策略': 'mpc',
        '固定大堂策略': 'lobby',
        '均匀分布策略': 'uniform',
        '需求导向策略': 'demand'
    }
    
    # 5. 在测试集上评估各策略
    print("\n" + "=" * 70)
    print("步骤2: 在测试集上评估策略 (使用真实呼叫数据)")
    print("=" * 70)
    
    results = {}
    for strategy_name, strategy_type in strategies.items():
        results[strategy_name] = evaluate_strategy_on_test_set(
            controller, analyzer, strategy_type, strategy_name
        )
    
    # 6. 分析结果
    analyze_results(results, analyzer)
    
    # 7. 生成建议
    generate_recommendations(results, analyzer, controller)
    
    return results, analyzer, controller


def analyze_results(results, analyzer):
    """分析仿真结果"""
    
    print("\n" + "=" * 70)
    print("测试集性能对比分析")
    print("=" * 70)
    
    # 汇总每个策略的性能
    summary = {}
    for strategy_name in results.keys():
        total_weighted_wait = 0
        total_calls = 0
        
        for period in results[strategy_name]:
            perf = results[strategy_name][period]
            total_weighted_wait += perf['avg_waiting_time'] * perf['total_calls']
            total_calls += perf['total_calls']
        
        avg_wait = total_weighted_wait / max(total_calls, 1)
        summary[strategy_name] = {
            'avg_waiting': avg_wait,
            'total_calls': total_calls
        }
    
    # 打印排名
    print("\n各策略测试集平均等待时间排名:")
    sorted_strategies = sorted(summary.items(), key=lambda x: x[1]['avg_waiting'])
    for i, (name, data) in enumerate(sorted_strategies, 1):
        print(f"  {i}. {name}: {data['avg_waiting']:.1f}秒 (测试呼叫{data['total_calls']}次)")
    
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
    ax1.set_title('各策略测试集平均等待时间对比')
    ax1.tick_params(axis='x', rotation=15)
    
    for bar, val in zip(bars, waits):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom')
    
    # 2. 各时间段对比
    ax2 = axes[0, 1]
    periods = list(analyzer.time_periods.keys())
    
    for i, strategy in enumerate(strategies[:3]):
        period_waits = [results[strategy][p]['avg_waiting_time'] for p in periods]
        ax2.plot(periods, period_waits, 'o-', label=strategy, color=colors[i])
    
    ax2.set_xlabel('时间段')
    ax2.set_ylabel('平均等待时间(秒)')
    ax2.set_title('各时间段测试集等待时间对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 训练集呼叫分布（用于学习）
    ax3 = axes[1, 0]
    if 'A' in analyzer.call_patterns:
        dist = analyzer.call_patterns['A']['weekday']['distribution']
        floors = np.arange(1, analyzer.num_floors + 1)
        ax3.bar(floors, dist, color='#2E86AB', alpha=0.7)
        ax3.set_xlabel('楼层')
        ax3.set_ylabel('呼叫占比')
        ax3.set_title('训练集: 早高峰(A段)各楼层呼叫分布')
        ax3.set_xticks(floors)
    
    # 4. 训练/测试数据量对比
    ax4 = axes[1, 1]
    train_count = len(analyzer.train_hall_calls) if analyzer.train_hall_calls is not None else 0
    test_count = len(analyzer.test_hall_calls) if analyzer.test_hall_calls is not None else 0
    
    ax4.bar(['训练集', '测试集'], [train_count, test_count], color=['#2E86AB', '#A23B72'])
    ax4.set_ylabel('呼叫记录数')
    ax4.set_title('训练集/测试集数据量')
    
    for i, (label, count) in enumerate([('训练集', train_count), ('测试集', test_count)]):
        ax4.text(i, count + 1000, f'{count:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('elevator_mpc_analysis_v3.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存为 'elevator_mpc_analysis_v3.png'")
    plt.show()


def generate_recommendations(results, analyzer, controller):
    """生成停车策略建议"""
    
    print("\n" + "=" * 70)
    print("停车策略建议 (基于测试集验证)")
    print("=" * 70)
    
    # 找出最佳策略
    best_results = {}
    for period in analyzer.time_periods.keys():
        best_wait = float('inf')
        best_strategy = None
        
        for strategy_name in results.keys():
            wait = results[strategy_name][period]['avg_waiting_time']
            if wait < best_wait:
                best_wait = wait
                best_strategy = strategy_name
        
        best_results[period] = {
            'strategy': best_strategy,
            'wait_time': best_wait,
            'positions': results[best_strategy][period]['final_positions']
        }
    
    for period, (start, end) in analyzer.time_periods.items():
        print(f"\n{period}段 ({start}-{end}):")
        
        best = best_results[period]
        print(f"  最佳策略: {best['strategy']}")
        print(f"  建议停车位置: {sorted(best['positions'])}")
        print(f"  测试集验证等待时间: {best['wait_time']:.1f}秒")
        
        # 高需求楼层（从训练集学习）
        if period in analyzer.call_patterns:
            dist = analyzer.call_patterns[period]['weekday']['distribution']
            top_floors = np.argsort(dist)[-5:][::-1] + 1
            print(f"  高需求楼层(训练集): {list(top_floors)}")


# 主程序
if __name__ == "__main__":
    results, analyzer, controller = run_analysis_and_simulation()
