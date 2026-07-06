'''
Description: 特定的初始化操作
'''

import random
import numpy as np
import copy
from typing import List, Dict, Any
from algorithm.benchmark_process import calculate_arrive_time, calculate_distance, calculate_wait_time
from algorithm.cal_std_index import find_slack_times, cal_stdi_from_slacks

def heuristic_initialization(data, distance_matrix, customer_count, capacity, top_ratio=0.2, bias_factor=2.0):
    """
    启发式初始化方法:
    1. 从仓库开始构建路径
    2. 对于每个路径，检查未分配客户是否满足约束
    3. 按等待时间排序符合条件的客户，在前20%中使用轮盘赌选择（距离近概率大）
    4. 当无法再添加客户时，开始新的路径
    5. 返回路径方案
    
    Args:
        top_ratio: 等待时间排序后考虑的前百分比客户，默认0.2(20%)
        bias_factor: 控制距离转化为概率时的偏好强度，值越大越偏好近距离客户
    
    Returns:
        基于启发式规则的路径方案
    """
    # 路径方案列表初始化
    route_plan = []
    
    # 未分配客户列表(1到customer_count)
    unassigned = list(range(1, customer_count + 1))
    
    # 循环直到所有客户被分配
    while unassigned:
        # 新路径从仓库(0)开始
        current_route = [0]
        current_load = 0
        current_time = 0
        
        # 构建当前路径
        while True:
            feasible_customers = []
            
            # 评估所有未分配客户
            for customer in unassigned:
                # 获取客户信息
                demand = data['DEMAND'][customer]
                ready_time = data['READY_TIME'][customer]
                due_date = data['DUE_DATE'][customer]
                service_time = data['SERVICE_TIME'][customer]
                
                # 检查容量约束
                if current_load + demand > capacity:
                    continue
                
                # 计算到达时间
                last_node = current_route[-1]
                travel_time = distance_matrix[last_node][customer]
                arrival_time = current_time + travel_time
                
                # 检查时间窗约束    
                if arrival_time > due_date:
                    continue
                
                # 计算等待时间
                wait_time = max(0, ready_time - arrival_time)
                distance = distance_matrix[last_node][customer]
                
                # 记录满足约束的客户
                feasible_customers.append((customer, distance, wait_time))
            
            # 当前路径无法继续添加客户
            if not feasible_customers:
                current_route.append(0)  # 返回仓库
                route_plan.append(current_route)
                break
            
            # 按等待时间升序排序
            feasible_customers.sort(key=lambda x: x[2])
            
            # 选择前top_ratio比例的客户
            candidates_count = max(1, int(len(feasible_customers) * top_ratio))
            candidates = feasible_customers[:candidates_count]
            
            # 使用轮盘赌选择客户（距离越近，被选中概率越高）
            # 计算距离的倒数作为选择概率基础
            fitness_values = [1.0/(candidate[1]**bias_factor + 0.1) for candidate in candidates]  # 加0.1避免除零错误
            total_fitness = sum(fitness_values)
            probabilities = [fitness/total_fitness for fitness in fitness_values]
            
            # 轮盘赌选择
            r = random.random()
            cumulative_prob = 0
            selected_idx = 0
            for idx, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    selected_idx = idx
                    break
            
            next_customer = candidates[selected_idx][0]
            
            # 更新路径和相关状态
            current_route.append(next_customer)
            unassigned.remove(next_customer)
            current_load += data['DEMAND'][next_customer]
            
            # 更新当前时间(考虑等待时间和服务时间)
            last_node = current_route[-2]
            travel_time = distance_matrix[last_node][next_customer]
            arrival_time = current_time + travel_time
            ready_time = data['READY_TIME'][next_customer]
            service_time = data['SERVICE_TIME'][next_customer]
            
            current_time = max(arrival_time, ready_time) + service_time
            
            # 如果所有客户都已分配，结束路径
            if not unassigned:
                current_route.append(0)
                route_plan.append(current_route)
                break
    
    return route_plan

def random_initialization(max_vehicle, customer_count):
    """
    随机初始化方法:
    1. 生成随机路径方案
    2. 确保每个路径从仓库(0)开始并返回仓库
    3. 检查容量约束和时间窗约束
    4. 返回路径方案
    
    Args:
        max_vehicle: 最大车辆数量
        customer_count: 客户数量

    Returns:
        基于随机初始化的路径方案
    """
    # 路径方案列表初始化
    route_plan = [] 
    
    # 随机排列所有客户
    customer_sequence = np.array(range(1, customer_count + 1))
    np.random.shuffle(customer_sequence)
    
    # 随机生成分割点
    max_routes = min(max_vehicle, customer_count)
    num_splits = random.randint(1, max_routes - 1)
    
    # 确保分割点不重复且有序
    split_points = sorted(random.sample(range(1, customer_count), num_splits))
    
    # 根据分割点将客户序列分割成多个路径
    current_route = [0]
    for split in split_points:
        route_plan.append([0] + customer_sequence[current_route[-1]:split].tolist() + [0])
        current_route = [split]
    
    # 添加最后一个路径
    route_plan.append([0] + customer_sequence[current_route[-1]:].tolist() + [0])

    return route_plan
    
            
def objective_wise_initialization(data,
                                   distance_matrix: np.ndarray,
                                   travel_time_matrix: np.ndarray,
                                   customer_count: int,
                                   capacity: int,
                                   objective_chosen: int,
                                   high_redundancy_threshold: float = None,
                                   feasible_check_item: List[str] = None) -> List[List[int]]:
    """
    基于目标的插入式初始化：
    1. 所有客户先标记为“未分配”。
    2. 每一步，在所有 (客户, 路径, 插入位置) 三元组中，选出插入后全局目标值增量最小的那一个，并执行该插入。
    3. 若没有任何可行的插入，则从未分配集中选取一个客户，开启一条新的路径 [0, customer, 0]。
    4. 重复直到所有客户分配完毕。

    Args:
        data: 客户属性字典 (DEMAND, READY_TIME, DUE_DATE, SERVICE_TIME 等)。
        distance_matrix: 节点间距离/时间矩阵。
        customer_count: 客户总数 (编号 1…customer_count)。
        capacity: 车辆容量。
        objective_chosen: 1=距离, 2=等待时间, 3=延迟时间, 4=冗余空间均匀度。
        high_redundancy_threshold: 用于计算冗余空间的阈值（objective 4）。
        feasible_check_item: 可行性检查项列表，传给 check_route_feasible。

    Returns:
        route_plan: 初始种群中的一条个体（路径方案）。
    """

    # 辅助函数：计算当前解的总目标值
    def total_objective(routes):
        if objective_chosen == 1:
            return sum(calculate_distance(distance_matrix, r) for r in routes)
        if objective_chosen == 2:
            return sum(calculate_wait_time(data, travel_time_matrix, r) for r in routes)
        if objective_chosen == 3:
            total = 0
            for r in routes:
                arrive = calculate_arrive_time(data, travel_time_matrix, r)
                for idx, node in enumerate(r):
                    if idx > 0 and arrive[idx] > data['DUE_DATE'][node]:
                        total += arrive[idx] - data['DUE_DATE'][node]
            return total
        if objective_chosen == 4:
            spaces = find_slack_times(routes, data, travel_time_matrix)
            return -cal_stdi_from_slacks(spaces, data)
        
    unassigned = list(range(1, customer_count + 1))
    # 随机打乱未分配客户顺序
    random.shuffle(unassigned)
    routes: List[List[int]] = []
    current_obj = 0.0

    while unassigned:
        best = None
        best_delta = float('inf')

        # 遍历所有未分配客户，在所有路径与插入位置上试验
        for cust in unassigned:
            # 如果还没有任何路径，先跳过，待下面开启新路径
            for ridx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    if not check_route_feasible(new_route, data, travel_time_matrix, capacity,
                                                check_feasible_item=feasible_check_item,
                                                time_tolerance=high_redundancy_threshold):
                        continue
                    # 构造新解、计算目标值增量
                    tmp_routes = copy.deepcopy(routes)
                    tmp_routes[ridx] = new_route
                    new_obj = total_objective(tmp_routes)
                    delta = new_obj - current_obj
                    if delta < best_delta:
                        best_delta = delta
                        best = ('insert', cust, ridx, pos, new_obj)

        if best:
            _, cust, ridx, pos, new_obj = best
            # 执行插入
            routes[ridx].insert(pos, cust)
            unassigned.remove(cust)
            current_obj = new_obj
        else:
            # 没有任何可行插入 —— 开新路径
            seed = unassigned.pop(0)
            new_route = [0, seed, 0]
            routes.append(new_route)
            # 更新目标值
            current_obj = total_objective(routes)

    return routes

def check_route_feasible(route: List[int], data: Dict[str, Any], travel_time_matrix: np.ndarray, capacity: int, check_feasible_item: List[str], time_tolerance: float=0) -> bool:
    '''
    检查路径是否可行
    :param route: 路径
    :param data: 客户数据
    :param travel_time_matrix: 旅行时间矩阵
    :param max_vehicle: 最大车辆数量
    :param capacity: 车辆容量
    :return: 是否可行
    '''
    if 'capacity' in check_feasible_item:
        # 检查路径容量
        load = 0
        for customer in route[1:]:
            load += data['DEMAND'][customer]
        if load > capacity:
            return False
    
    if 'time_window' in check_feasible_item:
        # 检查时间窗
        arrive_time = calculate_arrive_time(data, travel_time_matrix, route)
        for i in range(1, len(route)):
            if arrive_time[i] > data['DUE_DATE'][route[i]]:
                return False
            
    if 'time_window_soft' in check_feasible_item:
        # 检查时间窗
        arrive_time = calculate_arrive_time(data, travel_time_matrix, route)
        for i in range(1, len(route)):
            if arrive_time[i] > data['DUE_DATE'][route[i]] + 0 * time_tolerance:
                # 允许一定的时间宽容
                return False

    return True
