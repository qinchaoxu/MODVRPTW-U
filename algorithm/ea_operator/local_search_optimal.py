'''
Description: 特定的局部搜索操作(如车辆数量、总成本、总等待时间、总延迟时间、冗余时间均匀度指标等)的实现
最新版局部搜索，目标是冗余时间均匀度指标（所有客户的冗余）
'''

import numpy as np
import random
import copy
from typing import List, Dict, Any
from ..benchmark_process import calculate_arrive_time, calculate_distance, calculate_wait_time
from ..cal_std_index import find_slack_times, cal_stdi, cal_stdi_from_slacks


def objective_wise_local_search(solution: List[List[int]], data: Dict[str, Any], distance_matrix: np.ndarray, travel_time_matrix: np.ndarray, max_vehicle: int, capacity: int, high_redundancy_threshold: float,
                                 weight_vector: List[float], objective_index: List[int], feasible_check_item: List[str]) -> List[List[int]]:
    '''
    基于目标的局部搜索
    :param solution: 当前解
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
    :param travel_time_matrix: 旅行时间矩阵
    :param max_vehicle: 最大车辆数量
    :param capacity: 车辆容量
    :param weight_vector: 目标权重向量
    :param objective_index: 目标索引，（0：车辆数，1：总成本，2：总等待时间，3：总延迟时间，4：冗余空间均匀度指标）
    :param feasible_check_item: 可行性检查项，（"vehicle_num", "capacity", "time_window", "customer_visit"）
    
    :return: 局部搜索后的解
    '''
    # 根据目标权重向量，进行轮盘赌选择(权重向量本身作为概率)
    objective_index_chosen = np.random.choice(len(weight_vector), p=weight_vector)
    # 选择目标索引
    objective_chosen = objective_index[objective_index_chosen]
    # 选择目标索引对应的局部搜索操作
    if objective_chosen == 0:
        # 车辆数量局部搜索
        solution = vehicle_number_local_search(solution, data, distance_matrix, travel_time_matrix, capacity, high_redundancy_threshold, weight_vector, objective_index, feasible_check_item)
    else:
        # 其他目标的局部搜索
        solution = general_objective_local_search(solution, objective_chosen, data, distance_matrix, travel_time_matrix, max_vehicle, capacity, high_redundancy_threshold, feasible_check_item)
    return solution
    

def vehicle_number_local_search(solution: List[List[int]], data: Dict[str, Any], distance_matrix: np.ndarray, travel_time_matrix: np.ndarray, capacity: int, high_redundancy_threshold: float,
                                weight_vector: List[float], objective_index: List[int], feasible_check_item: List[str]) -> List[List[int]]:
    '''
    车辆数量局部搜索
    '''
    # 在权重向量中去除车辆路径，重新进行归一化
    new_weight_vector = weight_vector[1:].copy()
    s = np.sum(new_weight_vector)
    if s == 0:
        # 如果所有目标权重都为0，则把其余权重平均分配
        new_weight_vector = np.ones(len(new_weight_vector)) / len(new_weight_vector)
    else:
        # 归一化权重向量
        new_weight_vector /= s
    all_vector = [0, 0, 0, 0]
    for i in range(len(new_weight_vector)):
        all_vector[objective_index[i + 1] - 1] = new_weight_vector[i]

    # 选择路径的方法
    route_selection_method = random.choice(['min_customer', 'min_objective'])
    if route_selection_method == 'min_customer':
        # 选择客户数量最少的路径，将其客户依次插入其他路径中
        min_path_index = np.argmin([len(path) for path in solution])
        min_path = solution[min_path_index]
        customers_to_redistribute = [c for c in min_path if c != 0]
        customer_insert_fail = []
        # 从路径列表中移除目标路径
        new_routes = [route for i, route in enumerate(solution) if i != min_path_index]
        if not new_routes:
            # 如果没有其他路径，则返回原路径
            return solution
    else:
        # 根据新的权重选择一个objective
        objective_chosen = np.random.choice(len(all_vector), p=all_vector) + 1
        routes = copy.deepcopy(solution)
        # 选择路径
        if objective_chosen == 1:
            # 计算每条路径的单位成本
            route_cost_per_customer = [calculate_distance(distance_matrix, route)/len(route) for route in routes]
            # 选择单位成本最高的路径
            route_index_chosen = np.argmax(route_cost_per_customer)
        if objective_chosen == 2:
            # 计算每条路径的单位等待时间
            route_wait_time = [calculate_wait_time(data, travel_time_matrix, route) for route in routes]
            route_wait_time_per_customer = [time / len(route) for time, route in zip(route_wait_time, routes)]
            # 选择单位等待时间最高的路径
            route_index_chosen = np.argmax(route_wait_time_per_customer)
        if objective_chosen == 3:
            # 计算每条路径的总延迟时间
            delay_time = [0] * len(routes)
            for i, route in enumerate(routes):
                arrive_time = calculate_arrive_time(data, travel_time_matrix, route)
                for j in range(1, len(route)):
                    if arrive_time[j] > data['DUE_DATE'][route[j]]:
                        delay_time[i] += arrive_time[j] - data['DUE_DATE'][route[j]]
            delay_time_per_customer = [time / len(route) for time, route in zip(delay_time, routes)]
            # 选择单位延迟时间最高的路径
            route_index_chosen = np.argmax(delay_time_per_customer)
        if objective_chosen == 4:
            # 找到每条路径中的散布均匀度指标
            std_index_route = [cal_stdi([route], data, travel_time_matrix) for route in routes]
            # 选择散布均匀度指标最低的路径
            route_index_chosen = np.argmin(std_index_route)
            
        route_chosen = routes[route_index_chosen]
        customers_to_redistribute = [c for c in route_chosen if c != 0]
        customer_insert_fail = []
        # 从路径列表中移除目标路径
        new_routes = [route for i, route in enumerate(routes) if i != route_index_chosen]
        if not new_routes:
            # 如果没有其他路径，则返回原路径
            return solution

    route_cost = [0] * len(new_routes)  # 每条路径的总成本
    route_wait_time = [0] * len(new_routes)  # 每条路径的总等待时间
    delay_time = [0] * len(new_routes)  # 每条路径的总延迟时间
    redundancy_space_route = [0] * len(new_routes)  # 每条路径的冗余空间
    # 根据目标索引计算需要计算的目标
    all_objectives = {'total_cost': 0, 'total_wait_time': 0, 'total_delay_time': 0, 'redundancy_space_std': 0}
    if 1 in objective_index:
        # 计算每条路径的总成本
        route_cost = [calculate_distance(distance_matrix, route) for route in new_routes]
        all_objectives['total_cost'] = sum(route_cost)
    if 2 in objective_index:
        # 计算每条路径的总等待时间
        route_wait_time = [calculate_wait_time(data, travel_time_matrix, route) for route in new_routes]
        all_objectives['total_wait_time'] = sum(route_wait_time)
    if 3 in objective_index:
        # 计算每条路径的总延迟时间
        delay_time = [0] * len(new_routes)
        for i, route in enumerate(new_routes):
            arrive_time = calculate_arrive_time(data, travel_time_matrix, route)
            for j in range(1, len(route)):
                if arrive_time[j] > data['DUE_DATE'][route[j]]:
                    delay_time[i] += arrive_time[j] - data['DUE_DATE'][route[j]]
        all_objectives['total_delay_time'] = sum(delay_time)
    if 4 in objective_index:
        # 找到每条路径中的高冗余空间
        redundancy_space_route = find_slack_times(new_routes, data, travel_time_matrix)
        # 计算冗余空间均匀度指标
        all_objectives['redundancy_space_std'] = -cal_stdi_from_slacks(redundancy_space_route, data)

    current_objective_value = all_vector[0] * all_objectives['total_cost'] + \
                              all_vector[1] * all_objectives['total_wait_time'] + \
                              all_vector[2] * all_objectives['total_delay_time'] + \
                              all_vector[3] * all_objectives['redundancy_space_std']

    # 尝试将客户插入其他路径
    for customer in customers_to_redistribute:
        inserted = False
        # 按剩余容量排序路径
        metrics = list(zip(new_routes, 
                    route_cost, 
                    route_wait_time, 
                    delay_time, 
                    redundancy_space_route))
        metrics.sort(
                    key=lambda x: capacity - sum(data['DEMAND'][c] for c in x[0] if c != 0),
                    reverse=True)
        new_routes, route_cost, route_wait_time, delay_time, redundancy_space_route = map(list, zip(*metrics))

        min_objective_value = float('inf')
        accepted_move = False
        # 尝试插入到每条路径
        for i, route in enumerate(new_routes):
            # 检查容量约束
            route_demand = sum(data['DEMAND'][c] for c in route if c != 0)
            customer_demand = data['DEMAND'][customer]
            if route_demand + customer_demand > capacity:
                continue
            
            # 尝试将客户插入到路径中的每个位置
            # 找到加权目标函数值最小的路径
            for j in range(1, len(route)):
                new_route = route[:j] + [customer] + route[j:]  # 在路径中插入客户
                # 检查路径可行性
                if check_route_feasible(new_route, data, travel_time_matrix, capacity, check_feasible_item=feasible_check_item, time_tolerance=high_redundancy_threshold):
                    # 计算插入后的目标函数值
                    new_objectives = {'total_cost': 0, 'total_wait_time': 0, 'total_delay_time': 0, 'redundancy_space_std': 0}
                    if 1 in objective_index:
                        new_route_cost = calculate_distance(distance_matrix, new_route)  # 计算插入后这条路径的成本
                        new_route_cost_change = new_route_cost - route_cost[i]  # 计算插入后这条路径成本增加值
                        new_objectives['total_cost'] = all_objectives['total_cost'] + new_route_cost_change
                    if 2 in objective_index:
                        new_route_wait_time = calculate_wait_time(data, travel_time_matrix, new_route)  # 计算插入后这条路径的等待时间
                        new_route_wait_time_change = new_route_wait_time - route_wait_time[i]
                        new_objectives['total_wait_time'] = all_objectives['total_wait_time'] + new_route_wait_time_change
                    if 3 in objective_index:
                        new_route_delay_time = 0
                        new_route_arrive_time = calculate_arrive_time(data, travel_time_matrix, new_route)
                        for k in range(1, len(new_route)):
                            if new_route_arrive_time[k] > data['DUE_DATE'][new_route[k]]:
                                new_route_delay_time += new_route_arrive_time[k] - data['DUE_DATE'][new_route[k]]
                        new_route_delay_time_change = new_route_delay_time - delay_time[i]  # 计算插入后这条路径延迟时间增加值
                        new_objectives['total_delay_time'] = all_objectives['total_delay_time'] + new_route_delay_time_change
                    if 4 in objective_index:
                        # 计算插入后这条路径的冗余空间均匀度指标
                        new_route_redundancy_space = find_slack_times([new_route], data, travel_time_matrix)[0]
                        # 把新路径的冗余空间替换掉旧路径的冗余空间
                        redundancy_space_route_copy = redundancy_space_route.copy()
                        redundancy_space_route_copy[i] = new_route_redundancy_space
                        new_objectives['redundancy_space_std'] = -cal_stdi_from_slacks(redundancy_space_route_copy, data)

                    new_objective_value = all_vector[0] * new_objectives['total_cost'] +\
                                            all_vector[1] * new_objectives['total_wait_time'] + \
                                            all_vector[2] * new_objectives['total_delay_time'] + \
                                            all_vector[3] * new_objectives['redundancy_space_std']

                    # first-improvement：找到第一个改进解后立即接受
                    if new_objective_value < min_objective_value:
                        min_objective_value = new_objective_value
                        best_route = new_route
                        best_route_index = i
                        inserted = True
                        accepted_move = True

                        # 记录当前路径的目标函数中间值
                        if 1 in objective_index:
                            best_route_cost = new_route_cost
                            best_cost = new_objectives['total_cost']
                        if 2 in objective_index:
                            best_route_wait_time = new_route_wait_time
                            best_wait_time = new_objectives['total_wait_time']
                        if 3 in objective_index:
                            best_route_delay_time = new_route_delay_time
                            best_delay_time = new_objectives['total_delay_time']
                        if 4 in objective_index:
                            best_route_redundancy_space = new_route_redundancy_space
                            best_redundancy_space_std = new_objectives['redundancy_space_std']

                        break
            if accepted_move:
                break

        if inserted:
            new_routes[best_route_index] = best_route
            # 更新目标函数值
            if 1 in objective_index:
                all_objectives['total_cost'] = best_cost
                route_cost[best_route_index] = best_route_cost
            if 2 in objective_index:
                all_objectives['total_wait_time'] = best_wait_time
                route_wait_time[best_route_index] = best_route_wait_time
            if 3 in objective_index:
                all_objectives['total_delay_time'] = best_delay_time
                delay_time[best_route_index] = best_route_delay_time
            if 4 in objective_index:
                all_objectives['redundancy_space_std'] = best_redundancy_space_std
                redundancy_space_route[best_route_index] = best_route_redundancy_space
            current_objective_value = all_vector[0] * all_objectives['total_cost'] + \
                                      all_vector[1] * all_objectives['total_wait_time'] + \
                                      all_vector[2] * all_objectives['total_delay_time'] + \
                                      all_vector[3] * all_objectives['redundancy_space_std']

        if not inserted:
            # 如果没有找到可行位置，则将客户添加到未插入列表
            customer_insert_fail.append(customer)

    # 为未插入的客户创建新路径
    if customer_insert_fail:
        new_route = [0] + customer_insert_fail + [0]
        new_routes.append(new_route)

    return new_routes

def general_objective_local_search(solution: List[List[int]], objective_chosen: int, data: Dict[str, Any], distance_matrix: np.ndarray, travel_time_matrix: np.ndarray, max_vehicle: int, capacity: int, high_redundancy_threshold: float, feasible_check_item: List[str]) -> List[List[int]]:
    '''
    一般局部搜索操作，适用于车辆数量之外的其他目标
    :param solution: 当前解
    :param objective_chosen: 选择的目标
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
    :param max_vehicle: 最大车辆数量
    :param capacity: 车辆容量
    :param high_redundancy_threshold: 高冗余空间阈值
    :return: 更新后的解
    '''
    # 以下逻辑是依次执行所有局部搜索操作

    solution = general_objective_local_search_relocate(solution, objective_chosen, data, distance_matrix, travel_time_matrix, max_vehicle, capacity, high_redundancy_threshold, feasible_check_item)
    solution = general_objective_local_search_2opt(solution, objective_chosen, data, distance_matrix, travel_time_matrix, max_vehicle, capacity, high_redundancy_threshold, feasible_check_item)
    solution = general_objective_local_search_oropt(solution, objective_chosen, data, distance_matrix, travel_time_matrix, max_vehicle, capacity, high_redundancy_threshold, feasible_check_item)
    solution = general_objective_local_search_exchange(solution, objective_chosen, data, distance_matrix, travel_time_matrix, max_vehicle, capacity, high_redundancy_threshold, feasible_check_item)
    solution = general_objective_local_search_2opt_star(solution, objective_chosen, data, distance_matrix, travel_time_matrix, max_vehicle, capacity, high_redundancy_threshold, feasible_check_item)
    solution = general_objective_local_search_cross_exchange(solution, objective_chosen, data, distance_matrix, travel_time_matrix, max_vehicle, capacity, high_redundancy_threshold, feasible_check_item)


    # 以下实现随机选择
    # inter_route_ls_operator = random.choice([general_objective_local_search_relocate,
    #                                          general_objective_local_search_exchange, 
    #                                          general_objective_local_search_2opt_star,
    #                                          general_objective_local_search_cross_exchange])
    # solution = inter_route_ls_operator(solution, objective_chosen, data, distance_matrix, travel_time_matrix, max_vehicle, capacity, high_redundancy_threshold, feasible_check_item)

    # intra_route_ls_operator = random.choice([general_objective_local_search_2opt
    #                               , general_objective_local_search_oropt])
    # solution = intra_route_ls_operator(solution, objective_chosen, data, distance_matrix, travel_time_matrix, max_vehicle, capacity, high_redundancy_threshold, feasible_check_item)

    return solution


def general_objective_local_search_relocate(solution: List[List[int]], objective_chosen: int, data: Dict[str, Any], distance_matrix: np.ndarray, travel_time_matrix: np.ndarray, max_vehicle: int, capacity: int, high_redundancy_threshold: float, feasible_check_item: List[str]) -> List[List[int]]:
    '''
    针对一般目标的relocate局部搜索操作，适用于车辆数量之外的其他目标
    :param solution: 当前解
    :param objective_chosen: 选择的目标
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
    :param max_vehicle: 最大车辆数量
    :param capacity: 车辆容量
    :param high_redundancy_threshold: 高冗余空间阈值
    :return: 更新后的解
    '''
    routes = copy.deepcopy(solution)
    # # 随机选择一条路径，后续可以修改为按照一定规则选择
    route_index_chosen = random.randint(0, len(routes) - 1)
    route_chosen = routes[route_index_chosen]

    # 随机抽取一个或多个客户
    try:
        if len(route_chosen) >= 4 and random.random() >= 0.5:
            k = random.randint(2, len(route_chosen)-2)
        else:
            k = 1
        customer_index_chosen = sorted(random.sample(range(1, len(route_chosen)-1), k=k))
    except Exception as e:
        print(f"Error: {e}")
        print(k, route_chosen)

    # 移除选中的客户
    customers_chosen = [route_chosen[i] for i in customer_index_chosen]
    for i in sorted(customer_index_chosen, reverse=True):
        del route_chosen[i]

    # 如果路径移除客户后为空，则删除该路径
    if len(route_chosen) <= 2:
        routes.remove(route_chosen)

    customer_insert_fail = []

    # 计算当前路径的目标函数值
    # 根据目标索引计算需要计算的目标
    if objective_chosen == 1:
        # 计算每条路径的总成本
        route_cost = [calculate_distance(distance_matrix, route) for route in routes]
        objective_value = sum(route_cost)
    if objective_chosen == 2:
        # 计算每条路径的总等待时间
        route_wait_time = [calculate_wait_time(data, travel_time_matrix, route) for route in routes]
        objective_value = sum(route_wait_time)
    if objective_chosen == 3:
        # 计算每条路径的总延迟时间
        delay_time = [0] * len(routes)
        for i, route in enumerate(routes):
            arrive_time = calculate_arrive_time(data, travel_time_matrix, route)
            for j in range(1, len(route)):
                if arrive_time[j] > data['DUE_DATE'][route[j]]:
                    delay_time[i] += arrive_time[j] - data['DUE_DATE'][route[j]]
        objective_value = sum(delay_time)
    if objective_chosen == 4:
        # 找到每条路径中的冗余空间
        redundancy_space_route = find_slack_times(routes, data, travel_time_matrix)
        # 计算冗余空间均匀度指标
        objective_value = -cal_stdi_from_slacks(redundancy_space_route, data)

    # 插入到所有路径中的最佳位置（根据选择的目标函数）
    for customer in customers_chosen:
        best_route_index = -1
        best_route = None
        min_objective_value = float('inf')
        accepted_move = False
        inserted = False

        # 遍历所有路径，寻找最佳插入位置
        for i, route in enumerate(routes):
            
            # 尝试将客户插入到路径中的每个位置
            for j in range(1, len(route)):
                route_copy = route.copy()
                new_route = route_copy[:j] + [customer] + route_copy[j:]  # 在路径中插入客户
                # 检查路径可行性
                if check_route_feasible(new_route, data, travel_time_matrix, capacity, check_feasible_item=feasible_check_item,time_tolerance=high_redundancy_threshold):
                    # 计算插入后的目标函数值
                    if objective_chosen == 1:
                        new_route_cost = calculate_distance(distance_matrix, new_route)  # 计算插入后这条路径的成本
                        new_route_cost_change = new_route_cost - route_cost[i]  # 计算插入后这条路径成本增加值
                        new_objective_value = objective_value + new_route_cost_change
                    if objective_chosen == 2:
                        new_route_wait_time = calculate_wait_time(data, travel_time_matrix, new_route)  # 计算插入后这条路径的等待时间
                        new_route_wait_time_change = new_route_wait_time - route_wait_time[i]
                        new_objective_value = objective_value + new_route_wait_time_change
                    if objective_chosen == 3:
                        new_route_delay_time = 0
                        new_route_arrive_time = calculate_arrive_time(data, travel_time_matrix, new_route)
                        for k in range(1, len(new_route)):
                            if new_route_arrive_time[k] > data['DUE_DATE'][new_route[k]]:
                                new_route_delay_time += new_route_arrive_time[k] - data['DUE_DATE'][new_route[k]]
                        new_route_delay_time_change = new_route_delay_time - delay_time[i]  # 计算插入后这条路径延迟时间增加值
                        new_objective_value = objective_value + new_route_delay_time_change
                    if objective_chosen == 4:
                        # 计算插入后这条路径的冗余空间均匀度指标
                        new_route_redundancy_space = find_slack_times([new_route], data, travel_time_matrix)[0]
                        # 把新路径的冗余空间替换掉旧路径的冗余空间
                        redundancy_space_route_copy = redundancy_space_route.copy()
                        redundancy_space_route_copy[i] = new_route_redundancy_space
                        new_objective_value = -cal_stdi_from_slacks(redundancy_space_route_copy, data)

                    # first-improvement：找到第一个改进解后立即接受
                    if new_objective_value < min_objective_value:
                        min_objective_value = new_objective_value
                        best_route = new_route
                        best_route_index = i
                        inserted = True
                        accepted_move = True

                        # 记录当前路径的目标函数中间值
                        if objective_chosen == 1:
                            best_route_cost = new_route_cost
                        if objective_chosen == 2:
                            best_route_wait_time = new_route_wait_time
                        if objective_chosen == 3:
                            best_route_delay_time = new_route_delay_time
                        if objective_chosen == 4:
                            best_route_redundancy_space = new_route_redundancy_space
                        break
            if accepted_move:
                break

        if inserted:
            # 更新解
            routes[best_route_index] = best_route
            # 更新目标函数值
            objective_value = min_objective_value
            # 更新中间计算值
            if objective_chosen == 1:
                route_cost[best_route_index] = best_route_cost
            if objective_chosen == 2:
                route_wait_time[best_route_index] = best_route_wait_time
            if objective_chosen == 3:
                delay_time[best_route_index] = best_route_delay_time
            if objective_chosen == 4:
                redundancy_space_route[best_route_index] = best_route_redundancy_space
        else:
            # 如果没有找到可行位置，则将客户添加到未插入列表
            customer_insert_fail.append(customer)
            continue

    # 如果有客户未插入，则创建新路径
    if customer_insert_fail:
        new_route = [0] + customer_insert_fail + [0]
        routes.append(new_route)

    return routes  # 返回更新后的解

def general_objective_local_search_2opt(solution: List[List[int]], objective_chosen: int, data: Dict[str, Any], distance_matrix: np.ndarray, travel_time_matrix: np.ndarray, max_vehicle: int, capacity: int, high_redundancy_threshold: float, feasible_check_item: List[str]) -> List[List[int]]:
    '''
    针对一般目标的2-opt局部搜索操作，适用于车辆数量之外的其他目标
    :param solution: 当前解
    :param objective_chosen: 选择的目标
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
    :param max_vehicle: 最大车辆数量
    :param capacity: 车辆容量
    :param high_redundancy_threshold: 高冗余空间阈值
    :return: 更新后的解
    '''
    routes = copy.deepcopy(solution)

    # 计算当前路径的目标函数值
    # 根据目标索引计算需要计算的目标
    if objective_chosen == 1:
        # 计算每条路径的总成本
        route_cost = [calculate_distance(distance_matrix, route) for route in routes]
        objective_value = sum(route_cost)
    if objective_chosen == 2:
        # 计算每条路径的总等待时间
        route_wait_time = [calculate_wait_time(data, travel_time_matrix, route) for route in routes]
        objective_value = sum(route_wait_time)
    if objective_chosen == 3:
        # 计算每条路径的总延迟时间
        delay_time = [0] * len(routes)
        for i, route in enumerate(routes):
            arrive_time = calculate_arrive_time(data, travel_time_matrix, route)
            for j in range(1, len(route)):
                if arrive_time[j] > data['DUE_DATE'][route[j]]:
                    delay_time[i] += arrive_time[j] - data['DUE_DATE'][route[j]]
        objective_value = sum(delay_time)
    if objective_chosen == 4:
        # 找到每条路径中的冗余空间
        redundancy_space_route = find_slack_times(routes, data, travel_time_matrix)
        # 计算冗余空间均匀度指标
        objective_value = -cal_stdi_from_slacks(redundancy_space_route, data)

    for r_idx, route in enumerate(routes):

        if len(route) <= 3:
            continue

        best_route = route.copy()
        best_objective_value = objective_value
        route_changed = False # 记录路径是否发生变化
        accepted_move = False

        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                # 生成新的路径
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                # 检查路径可行性
                if check_route_feasible(new_route, data, travel_time_matrix, capacity, check_feasible_item=feasible_check_item, time_tolerance=high_redundancy_threshold):
                    # 计算插入后的目标函数值
                    if objective_chosen == 1:
                        new_route_cost = calculate_distance(distance_matrix, new_route)
                        new_route_cost_change = new_route_cost - route_cost[r_idx]
                        new_objective_value = objective_value + new_route_cost_change
                    if objective_chosen == 2:
                        new_route_wait_time = calculate_wait_time(data, travel_time_matrix, new_route)
                        new_route_wait_time_change = new_route_wait_time - route_wait_time[r_idx]
                        new_objective_value = objective_value + new_route_wait_time_change
                    if objective_chosen == 3:
                        new_route_delay_time = 0
                        new_route_arrive_time = calculate_arrive_time(data, travel_time_matrix, new_route)
                        for k in range(1, len(new_route)):
                            if new_route_arrive_time[k] > data['DUE_DATE'][new_route[k]]:
                                new_route_delay_time += new_route_arrive_time[k] - data['DUE_DATE'][new_route[k]]
                        new_route_delay_time_change = new_route_delay_time - delay_time[r_idx]
                        new_objective_value = objective_value + new_route_delay_time_change
                    if objective_chosen == 4:
                        new_route_redundancy_space = find_slack_times([new_route], data, travel_time_matrix)[0]
                        redundancy_space_route_copy = redundancy_space_route.copy()
                        redundancy_space_route_copy[r_idx] = new_route_redundancy_space
                        new_objective_value = -cal_stdi_from_slacks(redundancy_space_route_copy, data)
                    # 更新最小目标函数值
                    if new_objective_value < best_objective_value:
                        best_route = new_route
                        best_objective_value = new_objective_value
                        route_changed = True
                        accepted_move = True
                        # 记录当前路径的目标函数中间值
                        if objective_chosen == 1:
                            best_route_cost = new_route_cost
                        if objective_chosen == 2:
                            best_route_wait_time = new_route_wait_time
                        if objective_chosen == 3:
                            best_route_delay_time = new_route_delay_time
                        if objective_chosen == 4:
                            best_route_redundancy_space = new_route_redundancy_space
                        break
            if accepted_move:
                break
        # 如果路径发生变化，则更新路径
        if route_changed:
            # 更新路径
            routes[r_idx] = best_route
            # 更新目标函数值
            objective_value = best_objective_value
            # 更新中间计算值
            if objective_chosen == 1:
                route_cost[r_idx] = best_route_cost
            if objective_chosen == 2:
                route_wait_time[r_idx] = best_route_wait_time
            if objective_chosen == 3:
                delay_time[r_idx] = best_route_delay_time
            if objective_chosen == 4:
                redundancy_space_route[r_idx] = best_route_redundancy_space

    return routes  # 返回更新后的解

def general_objective_local_search_oropt(solution: List[List[int]], objective_chosen: int, data: Dict[str, Any], distance_matrix: np.ndarray, travel_time_matrix: np.ndarray, max_vehicle: int, capacity: int, high_redundancy_threshold: float, feasible_check_item: List[str]) -> List[List[int]]:
    """
    针对一般目标的 Or-OPT 片段重定位局部搜索操作，
    适用于车辆数量之外的其他目标

    :param solution: 当前解，每条路径以 0 开头和结束
    :param objective_chosen: 选择的目标 (1=距离,2=等待时间,3=延迟,4=高冗余均匀度)
    :param data: 客户数据，需包含 'DEMAND','DUE_DATE'
    :param distance_matrix: 节点间距离矩阵
    :param max_vehicle: 最大车辆数量 (本算子内不新增路径)
    :param capacity: 车辆容量
    :param high_redundancy_threshold: 高冗余空间阈值
    :return: 优化后的解
    """
    routes = copy.deepcopy(solution)

    # 1. 预先计算中间量 & 全局目标值
    if objective_chosen == 1:
        route_cost = [calculate_distance(distance_matrix, rt) for rt in routes]
        objective_value = sum(route_cost)
    elif objective_chosen == 2:
        route_wait_time = [calculate_wait_time(data, travel_time_matrix, rt) for rt in routes]
        objective_value = sum(route_wait_time)
    elif objective_chosen == 3:
        delay_time = []
        for rt in routes:
            arr = calculate_arrive_time(data, travel_time_matrix, rt)
            d = sum(max(0, arr[idx] - data['DUE_DATE'][rt[idx]]) for idx in range(1, len(rt)))
            delay_time.append(d)
        objective_value = sum(delay_time)
    else:  # objective_chosen == 4
        redundancy_space_route = find_slack_times(routes, data, travel_time_matrix)
        objective_value = -cal_stdi_from_slacks(
            redundancy_space_route, data
        )

    # 2. 对每条路径做 Or-OPT (k = 1,2,3)
    for r_idx, route in enumerate(routes):
        n = len(route)
        if n <= 4:
            # 至少要有一个可摘片段 + 两个端点
            continue

        best_route = route
        best_objective = objective_value
        route_changed = False
        accepted_move = False

        # 尝试不同长度的片段 k
        for k in (1, 2, 3):
            if n <= k + 2:
                break  # 不足以摘取 k 个客户

            # 在 route 中摘取 [i:i+k]
            for i in range(1, n - k - 1):
                segment = route[i : i + k]
                remainder = route[:i] + route[i + k :]

                # 在 remainder 中尝试各个插入点 j
                for j in range(1, len(remainder)):
                    new_route = remainder[:j] + segment + remainder[j:]

                    # 容量+时窗可行性
                    total_demand = sum(data['DEMAND'][c] for c in new_route if c != 0)
                    if total_demand > capacity:
                        continue
                    if not check_route_feasible(
                        new_route, data, travel_time_matrix, capacity,
                        check_feasible_item=feasible_check_item,
                        time_tolerance=high_redundancy_threshold
                    ):
                        continue

                    # 计算新的全局目标值增量
                    if objective_chosen == 1:
                        new_cost = calculate_distance(distance_matrix, new_route)
                        delta = new_cost - route_cost[r_idx]
                    elif objective_chosen == 2:
                        new_wt = calculate_wait_time(data, travel_time_matrix, new_route)
                        delta = new_wt - route_wait_time[r_idx]
                    elif objective_chosen == 3:
                        arr = calculate_arrive_time(data, travel_time_matrix, new_route)
                        new_delay = sum(
                            max(0, arr[idx] - data['DUE_DATE'][new_route[idx]])
                            for idx in range(1, len(new_route))
                        )
                        delta = new_delay - delay_time[r_idx]
                    else:  # objective_chosen == 4
                        new_route_redundancy_space = find_slack_times(
                            [new_route], data, travel_time_matrix
                        )[0]
                        redundancy_space_route_copy = redundancy_space_route.copy()
                        redundancy_space_route_copy[r_idx] = new_route_redundancy_space
                        new_obj = -cal_stdi_from_slacks(
                            redundancy_space_route_copy, data
                        )
                        delta = new_obj - objective_value

                    new_objective = objective_value + delta

                    # 记录最优改进
                    if new_objective < best_objective:
                        best_objective = new_objective
                        best_route = new_route
                        route_changed = True
                        accepted_move = True
                        break
                if accepted_move:
                    break
            if accepted_move:
                break

        # 如果这一条路有改进，则应用
        if route_changed:
            routes[r_idx] = best_route
            # 更新全局目标与中间量
            objective_value = best_objective
            if objective_chosen == 1:
                route_cost[r_idx] = calculate_distance(distance_matrix, best_route)
            elif objective_chosen == 2:
                route_wait_time[r_idx] = calculate_wait_time(data, travel_time_matrix, best_route)
            elif objective_chosen == 3:
                arr = calculate_arrive_time(data, travel_time_matrix, best_route)
                delay_time[r_idx] = sum(
                    max(0, arr[idx] - data['DUE_DATE'][best_route[idx]])
                    for idx in range(1, len(best_route))
                )
            else:
                redundancy_space_route[r_idx] = find_slack_times(
                    [best_route], data, travel_time_matrix
                )[0]

    return routes  # 返回更新后的解

def general_objective_local_search_exchange(
    solution: List[List[int]],
    objective_chosen: int,
    data: Dict[str, Any],
    distance_matrix: np.ndarray,
    travel_time_matrix: np.ndarray,
    max_vehicle: int,
    capacity: int,
    high_redundancy_threshold: float,
    feasible_check_item: List[str]
) -> List[List[int]]:
    """
    针对一般目标的 Exchange（路径间客户交换）局部搜索算子。

    :param solution: 当前解，每条路径以 0 开头和结束
    :param objective_chosen: 选择的目标 (1=距离, 2=等待时间, 3=延迟, 4=高冗余均匀度)
    :param data: 客户数据，需包含 'DEMAND','DUE_DATE'
    :param distance_matrix: 节点间距离矩阵
    :param max_vehicle: 最大车辆数量 (本算子内不新增路径)
    :param capacity: 车辆容量
    :param high_redundancy_threshold: 高冗余空间阈值
    :param feasible_check_item: 传给 check_route_feasible 的约束项列表
    :return: 优化后的解
    """
    # 1. 深拷贝解
    routes = copy.deepcopy(solution)
    K = len(routes)

    # 2. 预先计算中间量 & 全局目标值
    if objective_chosen == 1:
        route_cost = [calculate_distance(distance_matrix, rt) for rt in routes]
        objective_value = sum(route_cost)
    elif objective_chosen == 2:
        route_wait_time = [calculate_wait_time(data, travel_time_matrix, rt) for rt in routes]
        objective_value = sum(route_wait_time)
    elif objective_chosen == 3:
        delay_time = []
        for rt in routes:
            arr = calculate_arrive_time(data, travel_time_matrix, rt)
            d = sum(max(0, arr[idx] - data['DUE_DATE'][rt[idx]]) for idx in range(1, len(rt)))
            delay_time.append(d)
        objective_value = sum(delay_time)
    else:  # objective_chosen == 4
        redundancy_space_route = find_slack_times(routes, data, travel_time_matrix)
        objective_value = -cal_stdi_from_slacks(
            redundancy_space_route, data
        )

    best_obj = objective_value
    best_move = None  # 格式 (p, q, i, j)
    improved = False

    # 3. 枚举所有路径对 (p, q) 及交换位置 (i, j)
    for p in range(K):
        for q in range(p+1, K):
            rp, rq = routes[p], routes[q]
            np_, nq_ = len(rp), len(rq)
            if np_ <= 2 or nq_ <= 2:
                continue

            for i in range(1, np_-1):
                for j in range(1, nq_-1):
                    # 生成交换后的两条路径副本
                    new_rp, new_rq = rp.copy(), rq.copy()
                    new_rp[i], new_rq[j] = new_rq[j], new_rp[i]

                    # 容量检查
                    if sum(data['DEMAND'][c] for c in new_rp if c!=0) > capacity:
                        continue
                    if sum(data['DEMAND'][c] for c in new_rq if c!=0) > capacity:
                        continue
                    # 时窗等可行性检查
                    if not check_route_feasible(
                        new_rp, data, travel_time_matrix, capacity,
                        check_feasible_item=feasible_check_item,
                        time_tolerance=high_redundancy_threshold
                    ):
                        continue
                    if not check_route_feasible(
                        new_rq, data, travel_time_matrix, capacity,
                        check_feasible_item=feasible_check_item,
                        time_tolerance=high_redundancy_threshold
                    ):
                        continue

                    # 计算增量 delta
                    if objective_chosen == 1:
                        c_p = calculate_distance(distance_matrix, new_rp)
                        c_q = calculate_distance(distance_matrix, new_rq)
                        delta = (c_p + c_q) - (route_cost[p] + route_cost[q])
                    elif objective_chosen == 2:
                        wt_p = calculate_wait_time(data, travel_time_matrix, new_rp)
                        wt_q = calculate_wait_time(data, travel_time_matrix, new_rq)
                        delta = (wt_p + wt_q) - (route_wait_time[p] + route_wait_time[q])
                    elif objective_chosen == 3:
                        arr_p = calculate_arrive_time(data, travel_time_matrix, new_rp)
                        arr_q = calculate_arrive_time(data, travel_time_matrix, new_rq)
                        d_p = sum(max(0, arr_p[k] - data['DUE_DATE'][new_rp[k]]) for k in range(1, len(new_rp)))
                        d_q = sum(max(0, arr_q[k] - data['DUE_DATE'][new_rq[k]]) for k in range(1, len(new_rq)))
                        delta = (d_p + d_q) - (delay_time[p] + delay_time[q])
                    else:
                        sp_p = find_slack_times(
                            [new_rp], data, travel_time_matrix
                        )[0]
                        sp_q = find_slack_times(
                            [new_rq], data, travel_time_matrix
                        )[0]
                        space_copy = redundancy_space_route.copy()
                        space_copy[p], space_copy[q] = sp_p, sp_q
                        new_obj = -cal_stdi_from_slacks(
                            space_copy, data
                        )
                        delta = new_obj - objective_value

                    new_obj_total = objective_value + delta
                    if new_obj_total < best_obj:
                        best_obj = new_obj_total
                        best_move = (p, q, i, j)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            break

    # 4. 应用最优交换
    if best_move:
        p, q, i, j = best_move
        routes[p][i], routes[q][j] = routes[q][j], routes[p][i]
        # 更新中间量 & 全局目标
        objective_value = best_obj
        if objective_chosen == 1:
            route_cost[p] = calculate_distance(distance_matrix, routes[p])
            route_cost[q] = calculate_distance(distance_matrix, routes[q])
        elif objective_chosen == 2:
            route_wait_time[p] = calculate_wait_time(data, travel_time_matrix, routes[p])
            route_wait_time[q] = calculate_wait_time(data, travel_time_matrix, routes[q])
        elif objective_chosen == 3:
            arr_p = calculate_arrive_time(data, travel_time_matrix, routes[p])
            arr_q = calculate_arrive_time(data, travel_time_matrix, routes[q])
            delay_time[p] = sum(max(0, arr_p[k] - data['DUE_DATE'][routes[p][k]]) for k in range(1, len(routes[p])))
            delay_time[q] = sum(max(0, arr_q[k] - data['DUE_DATE'][routes[q][k]]) for k in range(1, len(routes[q])))
        else:
            redundancy_space_route[p] = find_slack_times(
                [routes[p]], data, travel_time_matrix
            )[0]
            redundancy_space_route[q] = find_slack_times(
                [routes[q]], data, travel_time_matrix
            )[0]

    return routes

def general_objective_local_search_2opt_star(
    solution: List[List[int]],
    objective_chosen: int,
    data: Dict[str, Any],
    distance_matrix: np.ndarray,
    travel_time_matrix: np.ndarray,
    max_vehicle: int,
    capacity: int,
    high_redundancy_threshold: float,
    feasible_check_item: List[str]
) -> List[List[int]]:
    """
    针对一般目标的 2-opt*（交叉交换尾段）局部搜索算子，
    仅在两条路径之间交换尾段。

    :param solution: 当前解，每条路径以 0 开头和结束
    :param objective_chosen: 选择的目标 (1=距离, 2=等待时间, 3=延迟, 4=高冗余均匀度)
    """
    # 1. 深拷贝解
    routes = copy.deepcopy(solution)
    K = len(routes)

    # 2. 预计算中间量 & 全局目标值
    if objective_chosen == 1:
        route_cost = [calculate_distance(distance_matrix, rt) for rt in routes]
        objective_value = sum(route_cost)
    elif objective_chosen == 2:
        route_wait_time = [calculate_wait_time(data, travel_time_matrix, rt) for rt in routes]
        objective_value = sum(route_wait_time)
    elif objective_chosen == 3:
        delay_time = []
        for rt in routes:
            arr = calculate_arrive_time(data, travel_time_matrix, rt)
            d = sum(max(0, arr[idx] - data['DUE_DATE'][rt[idx]]) for idx in range(1, len(rt)))
            delay_time.append(d)
        objective_value = sum(delay_time)
    else:  # objective_chosen == 4
        redundancy_space_route = find_slack_times(routes, data, travel_time_matrix)
        objective_value = -cal_stdi_from_slacks(
            redundancy_space_route, data
        )

    best_obj = objective_value
    best_move = None  # 格式 (p, q, i, j)
    improved = False

    # 3. 枚举所有路径对 p<q 及切点 i,j
    for p in range(K):
        for q in range(p+1, K):
            rp, rq = routes[p], routes[q]
            np_, nq_ = len(rp), len(rq)
            # 每条路径至少要留 depot + 一个客户 + depot
            if np_ <= 3 or nq_ <= 3:
                continue

            # i in [1, np_-2], j in [1, nq_-2]
            for i in range(1, np_-1):
                for j in range(1, nq_-1):
                    # 交换尾段：尾部从 i 开始，尾部从 j 开始
                    new_rp = rp[:i] + rq[j:]
                    new_rq = rq[:j] + rp[i:]

                    # 容量检查
                    if sum(data['DEMAND'][c] for c in new_rp if c!=0) > capacity:
                        continue
                    if sum(data['DEMAND'][c] for c in new_rq if c!=0) > capacity:
                        continue

                    # 可行性检查（时窗、其他）
                    if not check_route_feasible(
                        new_rp, data, travel_time_matrix, capacity,
                        check_feasible_item=feasible_check_item,
                        time_tolerance=high_redundancy_threshold
                    ):
                        continue
                    if not check_route_feasible(
                        new_rq, data, travel_time_matrix, capacity,
                        check_feasible_item=feasible_check_item,
                        time_tolerance=high_redundancy_threshold
                    ):
                        continue

                    # 计算增量 delta
                    if objective_chosen == 1:
                        c_p = calculate_distance(distance_matrix, new_rp)
                        c_q = calculate_distance(distance_matrix, new_rq)
                        delta = (c_p + c_q) - (route_cost[p] + route_cost[q])
                    elif objective_chosen == 2:
                        wt_p = calculate_wait_time(data, travel_time_matrix, new_rp)
                        wt_q = calculate_wait_time(data, travel_time_matrix, new_rq)
                        delta = (wt_p + wt_q) - (route_wait_time[p] + route_wait_time[q])
                    elif objective_chosen == 3:
                        arr_p = calculate_arrive_time(data, travel_time_matrix, new_rp)
                        arr_q = calculate_arrive_time(data, travel_time_matrix, new_rq)
                        d_p = sum(max(0, arr_p[k] - data['DUE_DATE'][new_rp[k]]) for k in range(1, len(new_rp)))
                        d_q = sum(max(0, arr_q[k] - data['DUE_DATE'][new_rq[k]]) for k in range(1, len(new_rq)))
                        delta = (d_p + d_q) - (delay_time[p] + delay_time[q])
                    else:
                        sp_p = find_slack_times(
                            [new_rp], data, travel_time_matrix
                        )[0]
                        sp_q = find_slack_times(
                            [new_rq], data, travel_time_matrix
                        )[0]
                        space_copy = redundancy_space_route.copy()
                        space_copy[p], space_copy[q] = sp_p, sp_q
                        new_obj = -cal_stdi_from_slacks(
                            space_copy, data
                        )
                        delta = new_obj - objective_value

                    new_objective = objective_value + delta
                    if new_objective < best_obj:
                        best_obj = new_objective
                        best_move = (p, q, i, j)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            break

    # 4. 应用最优尾段交换
    if best_move:
        p, q, i, j = best_move
        rp, rq = routes[p], routes[q]
        routes[p] = rp[:i] + rq[j:]
        routes[q] = rq[:j] + rp[i:]

        # 更新中间量 & 全局目标
        objective_value = best_obj
        if objective_chosen == 1:
            route_cost[p] = calculate_distance(distance_matrix, routes[p])
            route_cost[q] = calculate_distance(distance_matrix, routes[q])
        elif objective_chosen == 2:
            route_wait_time[p] = calculate_wait_time(data, travel_time_matrix, routes[p])
            route_wait_time[q] = calculate_wait_time(data, travel_time_matrix, routes[q])
        elif objective_chosen == 3:
            arr_p = calculate_arrive_time(data, travel_time_matrix, routes[p])
            arr_q = calculate_arrive_time(data, travel_time_matrix, routes[q])
            delay_time[p] = sum(max(0, arr_p[k] - data['DUE_DATE'][routes[p][k]]) for k in range(1, len(routes[p])))
            delay_time[q] = sum(max(0, arr_q[k] - data['DUE_DATE'][routes[q][k]]) for k in range(1, len(routes[q])))
        else:
            redundancy_space_route[p] = find_slack_times(
                [routes[p]], data, travel_time_matrix
            )[0]
            redundancy_space_route[q] = find_slack_times(
                [routes[q]], data, travel_time_matrix
            )[0]

    return routes

def general_objective_local_search_cross_exchange(
    solution: List[List[int]],
    objective_chosen: int,
    data: Dict[str, Any],
    distance_matrix: np.ndarray,
    travel_time_matrix: np.ndarray,
    max_vehicle: int,
    capacity: int,
    high_redundancy_threshold: float,
    feasible_check_item: List[str]
) -> List[List[int]]:
    """
    针对一般目标的 Cross-Exchange（跨路径片段交换）局部搜索算子。

    :param solution: 当前解，每条路径以 0 开头和结束
    :param objective_chosen: 选择的目标 (1=距离,2=等待时间,3=延迟,4=高冗余均匀度)
    :param data: 客户数据，含 'DEMAND','DUE_DATE'
    :param distance_matrix: 节点间距离矩阵
    :param max_vehicle: 最大车辆数量（本算子内不新增路径）
    :param capacity: 车辆容量
    :param high_redundancy_threshold: 高冗余空间阈值（作为 time_tolerance）
    :param feasible_check_item: 传给 check_route_feasible 的约束项
    :return: 优化后的解
    """
    # 深拷贝解
    routes = copy.deepcopy(solution)
    K = len(routes)

    # 预计算中间量 & 全局目标
    if objective_chosen == 1:
        route_cost = [calculate_distance(distance_matrix, rt) for rt in routes]
        objective_value = sum(route_cost)
    elif objective_chosen == 2:
        route_wait_time = [calculate_wait_time(data, travel_time_matrix, rt) for rt in routes]
        objective_value = sum(route_wait_time)
    elif objective_chosen == 3:
        delay_time = []
        for rt in routes:
            arr = calculate_arrive_time(data, travel_time_matrix, rt)
            d = sum(max(0, arr[idx] - data['DUE_DATE'][rt[idx]]) for idx in range(1, len(rt)))
            delay_time.append(d)
        objective_value = sum(delay_time)
    else:
        redundancy_space_route = find_slack_times(routes, data, travel_time_matrix)
        objective_value = -cal_stdi_from_slacks(
            redundancy_space_route, data
        )

    best_obj = objective_value
    best_move = None  # 存 (p, q, i, j, k)
    improved = False

    # 枚举路径对 p<q
    for p in range(K):
        for q in range(p+1, K):
            rp, rq = routes[p], routes[q]
            np_, nq_ = len(rp), len(rq)
            if np_ <= 3 or nq_ <= 3:
                continue

            # k 为交换片段长度
            for k in (1, 2, 3):
                if np_ <= k+1 or nq_ <= k+1:
                    break

                # 在 rp 上取片段 rp[i:i+k]
                for i in range(1, np_ - k):
                    seg_p = rp[i:i+k]
                    rem_p = rp[:i] + rp[i+k:]

                    # 在 rq 上取片段 rq[j:j+k]
                    for j in range(1, nq_ - k):
                        seg_q = rq[j:j+k]
                        rem_q = rq[:j] + rq[j+k:]

                        new_p = rem_p[:i] + seg_q + rem_p[i:]
                        new_q = rem_q[:j] + seg_p + rem_q[j:]

                        # 容量检查
                        if sum(data['DEMAND'][c] for c in new_p if c!=0) > capacity:
                            continue
                        if sum(data['DEMAND'][c] for c in new_q if c!=0) > capacity:
                            continue

                        # 可行性检查
                        if not check_route_feasible(
                            new_p, data, travel_time_matrix, capacity,
                            check_feasible_item=feasible_check_item,
                            time_tolerance=high_redundancy_threshold
                        ):
                            continue
                        if not check_route_feasible(
                            new_q, data, travel_time_matrix, capacity,
                            check_feasible_item=feasible_check_item,
                            time_tolerance=high_redundancy_threshold
                        ):
                            continue

                        # 计算增量 delta
                        if objective_chosen == 1:
                            c_p = calculate_distance(distance_matrix, new_p)
                            c_q = calculate_distance(distance_matrix, new_q)
                            delta = (c_p + c_q) - (route_cost[p] + route_cost[q])
                        elif objective_chosen == 2:
                            wt_p = calculate_wait_time(data, travel_time_matrix, new_p)
                            wt_q = calculate_wait_time(data, travel_time_matrix, new_q)
                            delta = (wt_p + wt_q) - (route_wait_time[p] + route_wait_time[q])
                        elif objective_chosen == 3:
                            arr_p = calculate_arrive_time(data, travel_time_matrix, new_p)
                            arr_q = calculate_arrive_time(data, travel_time_matrix, new_q)
                            d_p = sum(max(0, arr_p[idx] - data['DUE_DATE'][new_p[idx]]) for idx in range(1, len(new_p)))
                            d_q = sum(max(0, arr_q[idx] - data['DUE_DATE'][new_q[idx]]) for idx in range(1, len(new_q)))
                            delta = (d_p + d_q) - (delay_time[p] + delay_time[q])
                        else:
                            sp_p = find_slack_times(
                                [new_p], data, travel_time_matrix
                            )[0]
                            sp_q = find_slack_times(
                                [new_q], data, travel_time_matrix
                            )[0]
                            space_copy = redundancy_space_route.copy()
                            space_copy[p], space_copy[q] = sp_p, sp_q
                            new_obj = -cal_stdi_from_slacks(
                                space_copy, data
                            )
                            delta = new_obj - objective_value

                        new_obj_total = objective_value + delta
                        if new_obj_total < best_obj:
                            best_obj = new_obj_total
                            best_move = (p, q, i, j, k)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            break

    # 应用最优跨路径片段交换
    if best_move:
        p, q, i, j, k = best_move
        rp, rq = routes[p], routes[q]
        seg_p = rp[i:i+k]
        seg_q = rq[j:j+k]
        rem_p = rp[:i] + rp[i+k:]
        rem_q = rq[:j] + rq[j+k:]
        routes[p] = rem_p[:i] + seg_q + rem_p[i:]
        routes[q] = rem_q[:j] + seg_p + rem_q[j:]

        # 更新中间量 & 全局目标
        objective_value = best_obj
        if objective_chosen == 1:
            route_cost[p] = calculate_distance(distance_matrix, routes[p])
            route_cost[q] = calculate_distance(distance_matrix, routes[q])
        elif objective_chosen == 2:
            route_wait_time[p] = calculate_wait_time(data, travel_time_matrix, routes[p])
            route_wait_time[q] = calculate_wait_time(data, travel_time_matrix, routes[q])
        elif objective_chosen == 3:
            arr_p = calculate_arrive_time(data, travel_time_matrix, routes[p])
            arr_q = calculate_arrive_time(data, travel_time_matrix, routes[q])
            delay_time[p] = sum(max(0, arr_p[k] - data['DUE_DATE'][routes[p][k]]) for k in range(1, len(routes[p])))
            delay_time[q] = sum(max(0, arr_q[k] - data['DUE_DATE'][routes[q][k]]) for k in range(1, len(routes[q])))
        else:
            redundancy_space_route[p] = find_slack_times(
                [routes[p]], data, travel_time_matrix
            )[0]
            redundancy_space_route[q] = find_slack_times(
                [routes[q]], data, travel_time_matrix
            )[0]

    return routes



def general_local_search(solution: List[List[int]], objective_chosen: int, data: Dict[str, Any], distance_matrix: np.ndarray, travel_time_matrix: np.ndarray, max_vehicle: int, capacity: int) -> List[List[int]]:
    '''
    一般局部搜索操作，采用常规的局部搜索算子
    :param solution: 当前解
    :param objective_chosen: 选择的目标
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
    :param max_vehicle: 最大车辆数量
    :param capacity: 车辆容量
    :return: 更新后的解
    '''
    # 这里可以添加其他局部搜索操作的实现
    # 例如，执行2-opt操作或其他基于目标的搜索
    return solution  # 返回更新后的解


def check_route_feasible(route: List[int], data: Dict[str, Any], travel_time_matrix: np.ndarray, capacity: int, check_feasible_item: List[str], time_tolerance: float=0) -> bool:
    '''
    检查路径是否可行
    :param route: 路径
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
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

