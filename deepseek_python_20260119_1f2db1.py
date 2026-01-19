import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import heapq
from collections import defaultdict, deque
import random
from typing import List, Dict, Tuple, Optional
import math

class Elevator:
    """电梯类"""
    def __init__(self, elevator_id: int, capacity: int = 20, current_floor: int = 1):
        self.id = elevator_id
        self.capacity = capacity  # 最大载客量
        self.current_floor = current_floor
        self.target_floors = []  # 目标楼层队列
        self.direction = 0  # 0:停止, 1:上行, -1:下行
        self.passengers = []  # 当前乘客
        self.load = 0  # 当前负载（人数）
        self.status = 'idle'  # idle, moving, loading, unloading
        self.waiting_time = 0  # 空闲等待时间
        self.total_distance = 0  # 总行驶距离
        self.last_update_time = datetime.now()
        
    def update_status(self, current_time):
        """更新电梯状态"""
        if self.status == 'idle':
            self.waiting_time += (current_time - self.last_update_time).seconds
        self.last_update_time = current_time
    
    def add_passenger(self, passenger):
        """添加乘客"""
        if self.load < self.capacity:
            self.passengers.append(passenger)
            self.load += 1
            # 添加目标楼层（如果不在列表中）
            if passenger.target_floor not in self.target_floors:
                self.target_floors.append(passenger.target_floor)
                # 按方向排序目标楼层
                self._sort_target_floors()
            return True
        return False
    
    def remove_passenger(self, floor):
        """移除到达目的地的乘客"""
        passengers_to_remove = [p for p in self.passengers if p.target_floor == floor]
        for p in passengers_to_remove:
            self.passengers.remove(p)
            self.load -= 1
        return len(passengers_to_remove)
    
    def _sort_target_floors(self):
        """按电梯方向排序目标楼层"""
        if self.direction == 1:  # 上行
            self.target_floors.sort()
        elif self.direction == -1:  # 下行
            self.target_floors.sort(reverse=True)
    
    def move_to_next_floor(self):
        """移动到下一层"""
        if not self.target_floors:
            self.direction = 0
            self.status = 'idle'
            return
        
        next_floor = self.target_floors[0]
        if next_floor > self.current_floor:
            self.direction = 1
            self.current_floor += 1
        elif next_floor < self.current_floor:
            self.direction = -1
            self.current_floor -= 1
        else:
            # 到达目标楼层
            self.target_floors.pop(0)
            self.status = 'unloading'
            self.total_distance += 1
        
        if self.target_floors:
            self.status = 'moving'
        else:
            self.status = 'idle'
            self.direction = 0

class Passenger:
    """乘客类"""
    def __init__(self, passenger_id: int, start_floor: int, target_floor: int, 
                 arrival_time: datetime):
        self.id = passenger_id
        self.start_floor = start_floor
        self.target_floor = target_floor
        self.arrival_time = arrival_time
        self.wait_start_time = arrival_time
        self.pickup_time = None
        self.dropoff_time = None
        self.waiting_time = 0  # 等待时间
        self.travel_time = 0  # 乘坐时间
    
    def calculate_waiting_time(self, pickup_time):
        """计算等待时间"""
        self.pickup_time = pickup_time
        self.waiting_time = (pickup_time - self.wait_start_time).seconds

class ElevatorSystem:
    """电梯系统类"""
    def __init__(self, num_elevators: int = 8, num_floors: int = 50, 
                 elevator_capacity: int = 20):
        self.num_floors = num_floors
        self.elevators = [Elevator(i, elevator_capacity, 1) for i in range(num_elevators)]
        self.waiting_passengers = defaultdict(list)  # 每层等待的乘客
        self.completed_passengers = []  # 已完成的乘客
        self.time = datetime.now()
        
        # 优化权重系数（可调整）
        self.weights = {
            'distance': 0.4,      # 位移权重
            'path': 0.3,          # 路径简洁度权重
            'load': 0.3           # 负载权重
        }
        
        # 交通模式识别
        self.traffic_patterns = {
            'morning_peak': {'start': 7, 'end': 9, 'direction': 'up'},
            'lunch_peak': {'start': 12, 'end': 13, 'direction': 'mixed'},
            'evening_peak': {'start': 17, 'end': 19, 'direction': 'down'},
            'off_peak': {'start': 0, 'end': 24, 'direction': 'mixed'}
        }
        
        # 历史数据统计
        self.history_data = {
            'floor_demand': defaultdict(list),  # 楼层需求历史
            'wait_times': [],  # 等待时间历史
            'energy_usage': []  # 能耗历史
        }
        
        # 预测模型参数
        self.prediction_window = 5 * 60  # 5分钟预测窗口（秒）
    
    def get_current_traffic_pattern(self):
        """获取当前交通模式"""
        current_hour = self.time.hour
        
        for pattern_name, pattern in self.traffic_patterns.items():
            if pattern_name == 'off_peak':
                continue
            if pattern['start'] <= current_hour < pattern['end']:
                return pattern_name, pattern
        
        return 'off_peak', self.traffic_patterns['off_peak']
    
    def predict_passenger_flow(self):
        """预测未来5分钟乘客流量"""
        # 基于历史数据的简单预测（可使用更复杂的模型如ARIMA、LSTM等）
        predictions = {}
        
        # 获取当前交通模式
        pattern_name, pattern = self.get_current_traffic_pattern()
        
        # 根据模式预测各楼层需求
        for floor in range(1, self.num_floors + 1):
            if pattern['direction'] == 'up':
                if floor == 1:  # 大堂上行需求高
                    predictions[floor] = random.uniform(5, 15)
                else:
                    predictions[floor] = random.uniform(0, 3)
            elif pattern['direction'] == 'down':
                if floor > 1:  # 上层下行需求高
                    predictions[floor] = random.uniform(3, 10)
                else:
                    predictions[floor] = random.uniform(0, 2)
            else:  # mixed
                predictions[floor] = random.uniform(1, 5)
        
        return predictions
    
    def calculate_elevator_score(self, elevator: Elevator, target_floor: int, 
                                 predicted_demand: Dict[int, float]) -> float:
        """计算电梯调度得分"""
        
        # 1. 最短位移分数（距离越近，分数越高）
        distance = abs(elevator.current_floor - target_floor)
        distance_score = 1.0 / (distance + 1)  # 避免除以零
        
        # 2. 最小路径分数（路径越简洁，分数越高）
        path_score = self.calculate_path_score(elevator, target_floor, predicted_demand)
        
        # 3. 最大负荷分数（预期需求越高，分数越高）
        demand = predicted_demand.get(target_floor, 0)
        load_score = min(demand / elevator.capacity, 1.0)
        
        # 加权总分
        total_score = (
            self.weights['distance'] * distance_score +
            self.weights['path'] * path_score +
            self.weights['load'] * load_score
        )
        
        return total_score
    
    def calculate_path_score(self, elevator: Elevator, target_floor: int,
                            predicted_demand: Dict[int, float]) -> float:
        """计算路径简洁度分数"""
        current_floor = elevator.current_floor
        
        # 如果电梯当前有乘客，考虑当前目标楼层
        if elevator.target_floors:
            # 计算从当前位置到目标楼层，再到预期高需求楼层的路径效率
            path_efficiency = 1.0
            
            # 考虑电梯当前方向
            if elevator.direction == 1:  # 上行
                if target_floor >= current_floor:
                    # 顺路，得分高
                    path_efficiency = 0.9
                else:
                    # 需要改变方向，得分低
                    path_efficiency = 0.4
            elif elevator.direction == -1:  # 下行
                if target_floor <= current_floor:
                    path_efficiency = 0.9
                else:
                    path_efficiency = 0.4
            else:  # 停止
                path_efficiency = 0.7
        else:
            # 空闲电梯
            path_efficiency = 0.8
        
        return path_efficiency
    
    def assign_elevator_to_passenger(self, passenger: Passenger) -> Optional[Elevator]:
        """为乘客分配电梯"""
        best_elevator = None
        best_score = -float('inf')
        
        # 预测未来需求
        predicted_demand = self.predict_passenger_flow()
        
        for elevator in self.elevators:
            # 检查电梯是否可用
            if elevator.load >= elevator.capacity:
                continue
                
            # 计算电梯接该乘客的得分
            score = self.calculate_elevator_score(elevator, passenger.start_floor, predicted_demand)
            
            # 调整得分：如果电梯方向与乘客方向一致，加分
            if elevator.target_floors:
                if passenger.target_floor > passenger.start_floor and elevator.direction == 1:
                    score *= 1.2
                elif passenger.target_floor < passenger.start_floor and elevator.direction == -1:
                    score *= 1.2
            
            if score > best_score:
                best_score = score
                best_elevator = elevator
        
        if best_elevator:
            # 添加乘客到电梯
            if best_elevator.add_passenger(passenger):
                # 添加起始楼层为目标（如果不在列表中）
                if passenger.start_floor not in best_elevator.target_floors:
                    best_elevator.target_floors.append(passenger.start_floor)
                    best_elevator._sort_target_floors()
                
                # 更新电梯状态
                if best_elevator.status == 'idle':
                    best_elevator.status = 'moving'
                    # 设置方向
                    if passenger.start_floor > best_elevator.current_floor:
                        best_elevator.direction = 1
                    elif passenger.start_floor < best_elevator.current_floor:
                        best_elevator.direction = -1
                
                return best_elevator
        
        return None
    
    def dynamic_parking_strategy(self):
        """动态停车策略"""
        # 获取当前交通模式
        pattern_name, pattern = self.get_current_traffic_pattern()
        
        # 预测未来需求
        predicted_demand = self.predict_passenger_flow()
        
        # 为每个空闲电梯选择停车楼层
        for elevator in self.elevators:
            if elevator.status == 'idle' and not elevator.target_floors:
                # 计算最佳停车楼层
                best_floor = 1  # 默认大堂
                best_score = -float('inf')
                
                # 根据交通模式调整候选楼层
                candidate_floors = []
                
                if pattern['direction'] == 'up':
                    # 早高峰：优先停在大堂和低楼层
                    candidate_floors = list(range(1, min(10, self.num_floors + 1)))
                elif pattern['direction'] == 'down':
                    # 晚高峰：优先停在高楼层
                    candidate_floors = list(range(max(1, self.num_floors - 9), self.num_floors + 1))
                else:
                    # 混合模式：考虑所有楼层
                    candidate_floors = list(range(1, self.num_floors + 1))
                
                # 为每个候选楼层评分
                for floor in candidate_floors:
                    score = self.calculate_elevator_score(elevator, floor, predicted_demand)
                    
                    # 避免多个电梯停在同层（简单协调）
                    parked_count = sum(1 for e in self.elevators 
                                     if e.current_floor == floor and e.status == 'idle')
                    if parked_count > 0:
                        score *= 0.7  # 降低分数
                    
                    if score > best_score:
                        best_score = score
                        best_floor = floor
                
                # 设置电梯目标楼层
                if best_floor != elevator.current_floor:
                    elevator.target_floors = [best_floor]
                    elevator.status = 'moving'
                    if best_floor > elevator.current_floor:
                        elevator.direction = 1
                    else:
                        elevator.direction = -1
    
    def simulate_time_step(self, seconds: int = 1):
        """模拟一个时间步长"""
        self.time += timedelta(seconds=seconds)
        
        # 更新所有电梯状态
        for elevator in self.elevators:
            elevator.update_status(self.time)
            
            # 处理到达目标楼层的电梯
            if elevator.target_floors and elevator.current_floor == elevator.target_floors[0]:
                # 卸载乘客
                passengers_unloaded = elevator.remove_passenger(elevator.current_floor)
                
                # 装载等待的乘客
                waiting_at_floor = self.waiting_passengers.get(elevator.current_floor, [])
                for passenger in waiting_at_floor[:]:
                    if elevator.add_passenger(passenger):
                        waiting_at_floor.remove(passenger)
                        passenger.pickup_time = self.time
                        passenger.calculate_waiting_time(self.time)
                
                # 移除已到达的目标楼层
                elevator.target_floors.pop(0)
                
                # 如果没有目标了，考虑动态停车
                if not elevator.target_floors:
                    self.dynamic_parking_strategy()
            
            # 移动电梯
            elevator.move_to_next_floor()
        
        # 定期执行动态停车策略（例如每30秒）
        if self.time.second % 30 == 0:
            self.dynamic_parking_strategy()
    
    def generate_passengers(self, num_passengers: int = 10):
        """生成模拟乘客"""
        pattern_name, pattern = self.get_current_traffic_pattern()
        
        for i in range(num_passengers):
            # 根据交通模式生成乘客
            if pattern['direction'] == 'up':
                start_floor = 1  # 从大堂开始
                target_floor = random.randint(2, self.num_floors)
            elif pattern['direction'] == 'down':
                start_floor = random.randint(2, self.num_floors)
                target_floor = 1  # 到大堂
            else:
                start_floor = random.randint(1, self.num_floors)
                target_floor = random.randint(1, self.num_floors)
                while target_floor == start_floor:
                    target_floor = random.randint(1, self.num_floors)
            
            passenger = Passenger(
                passenger_id=i,
                start_floor=start_floor,
                target_floor=target_floor,
                arrival_time=self.time
            )
            
            # 添加到等待列表
            self.waiting_passengers[start_floor].append(passenger)
            
            # 尝试分配电梯
            self.assign_elevator_to_passenger(passenger)
    
    def get_system_metrics(self):
        """获取系统性能指标"""
        total_wait_time = 0
        total_passengers = len(self.completed_passengers)
        
        for passenger in self.completed_passengers:
            total_wait_time += passenger.waiting_time
        
        avg_wait_time = total_wait_time / total_passengers if total_passengers > 0 else 0
        
        # 计算能耗（与行驶距离成正比）
        total_distance = sum(elevator.total_distance for elevator in self.elevators)
        
        # 计算电梯利用率
        idle_count = sum(1 for e in self.elevators if e.status == 'idle')
        utilization = 1 - (idle_count / len(self.elevators))
        
        return {
            'average_wait_time': avg_wait_time,
            'total_passengers': total_passengers,
            'total_distance': total_distance,
            'elevator_utilization': utilization,
            'current_time': self.time
        }
    
    def optimize_weights(self, iterations: int = 100):
        """优化权重系数（使用简单网格搜索）"""
        best_weights = self.weights.copy()
        best_score = float('inf')
        
        # 权重搜索空间
        weight_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for w1 in weight_values:
            for w2 in weight_values:
                for w3 in weight_values:
                    if abs(w1 + w2 + w3 - 1.0) > 0.01:  # 权重和需要接近1
                        continue
                    
                    self.weights = {'distance': w1, 'path': w2, 'load': w3}
                    
                    # 运行模拟测试
                    test_score = self.run_test_simulation(duration=600)  # 10分钟测试
                    
                    if test_score < best_score:
                        best_score = test_score
                        best_weights = self.weights.copy()
        
        self.weights = best_weights
        return best_weights, best_score
    
    def run_test_simulation(self, duration: int = 600) -> float:
        """运行测试模拟，返回综合得分（越低越好）"""
        # 保存当前状态
        original_time = self.time
        original_elevators = [Elevator(e.id, e.capacity, e.current_floor) for e in self.elevators]
        
        # 重置等待乘客和完成乘客
        original_waiting = self.waiting_passengers.copy()
        original_completed = self.completed_passengers.copy()
        
        # 运行模拟
        for _ in range(duration):
            # 随机生成乘客
            if random.random() < 0.3:  # 30%的概率生成新乘客
                self.generate_passengers(random.randint(1, 3))
            
            self.simulate_time_step()
        
        # 获取指标
        metrics = self.get_system_metrics()
        
        # 计算综合得分（加权平均等待时间和能耗）
        score = (
            0.7 * metrics['average_wait_time'] +  # 等待时间权重70%
            0.3 * (metrics['total_distance'] / 100)  # 能耗权重30%
        )
        
        # 恢复原始状态
        self.time = original_time
        self.elevators = original_elevators
        self.waiting_passengers = original_waiting
        self.completed_passengers = original_completed
        
        return score

def main():
    """主函数：演示电梯系统运行"""
    print("初始化电梯系统...")
    system = ElevatorSystem(num_elevators=8, num_floors=30, elevator_capacity=20)
    
    print("\n优化权重系数...")
    best_weights, best_score = system.optimize_weights(iterations=50)
    print(f"最佳权重: {best_weights}")
    print(f"最佳得分: {best_score:.2f}")
    
    print("\n开始模拟运行...")
    simulation_duration = 3600  # 模拟1小时
    
    for i in range(simulation_duration):
        # 每小时生成乘客数量变化
        if i % 60 == 0:  # 每分钟
            # 根据时间生成不同数量的乘客
            current_hour = system.time.hour
            if 7 <= current_hour < 9:  # 早高峰
                num_passengers = random.randint(5, 10)
            elif 12 <= current_hour < 13:  # 午间高峰
                num_passengers = random.randint(3, 8)
            elif 17 <= current_hour < 19:  # 晚高峰
                num_passengers = random.randint(6, 12)
            else:  # 非高峰
                num_passengers = random.randint(1, 4)
            
            system.generate_passengers(num_passengers)
        
        system.simulate_time_step()
        
        # 每小时输出一次状态
        if i % 3600 == 0 and i > 0:
            metrics = system.get_system_metrics()
            print(f"\n--- 运行 {i//3600} 小时报告 ---")
            print(f"平均等待时间: {metrics['average_wait_time']:.2f}秒")
            print(f"总乘客数: {metrics['total_passengers']}")
            print(f"总行驶距离: {metrics['total_distance']}层")
            print(f"电梯利用率: {metrics['elevator_utilization']:.2%}")
            
            # 显示电梯状态
            print("\n电梯状态:")
            for elevator in system.elevators:
                print(f"  电梯{elevator.id}: 楼层{elevator.current_floor}, "
                      f"状态{elevator.status}, 负载{elevator.load}/{elevator.capacity}")
    
    print("\n模拟完成!")
    
    # 最终报告
    final_metrics = system.get_system_metrics()
    print("\n=== 最终性能报告 ===")
    print(f"模拟时长: {simulation_duration//3600}小时")
    print(f"总服务乘客: {final_metrics['total_passengers']}")
    print(f"平均等待时间: {final_metrics['average_wait_time']:.2f}秒")
    print(f"总能耗(距离): {final_metrics['total_distance']}层")
    
    # 计算与传统策略的改进
    # （这里可以添加与基准策略的对比）

if __name__ == "__main__":
    main()