'''
Description: 特定的局部搜索操作(如车辆数量、总成本、总等待时间、总延迟时间、高冗余空间均匀度指标等)的实现
'''

import numpy as np
import random
import copy
from typing import List, Dict, Any
from copy import deepcopy
import time

def neighborhood_search_robust(vehicle_data, problem_data, distance_matrix, travel_time_disturbance=0, demand_disturbance=0, travel_time_matrix=None, iteration_max=100, p_kr=0.2):
    '''
    应用邻域搜索
    :param vehicle_data: 当前车辆数据
    :param problem_data: 问题数据
    :param distance_matrix: 距离矩阵
    :param travel_time_disturbance: 旅行时间扰动
    :param demand_disturbance: 需求扰动
    :param travel_time_matrix: 旅行时间矩阵
    '''
    if travel_time_matrix is None:
        # 如果没有提供旅行时间矩阵，则使用距离矩阵
        travel_time_matrix = distance_matrix
    start_time = time.time()

    vehicle_data = vnd(vehicle_data, problem_data, distance_matrix, travel_time_matrix, travel_time_disturbance, demand_disturbance, iteration_max, p_kr)

    end_time = time.time()
    # print(f'邻域搜索耗时: {end_time - start_time:.6f}秒')
    return vehicle_data

def vnd(
    vehicle_data,
    problem_data,
    distance_matrix,
    travel_time_matrix,
    travel_time_disturbance=0,
    demand_disturbance=0,
    iteration_max=100,
    p_kr=0.2
):
    """
    KRVND 主流程。按论文中的 6 个 First-Improvement 邻域算子顺序处理；
    一旦接受改进，下一轮从第一个算子重新开始。

    算子顺序：
    1. intra_route_2opt_first_improve
    2. intra_route_oropt_first_improve
    3. relocate_first_improve
    4. swap_first_improve
    5. inter_route_2opt_star_first_improve
    6. inter_route_cross_exchange_first_improve
    """
    neighborhoods = [
        intra_route_2opt_first_improve,
        intra_route_oropt_first_improve,
        relocate_first_improve,
        swap_first_improve,
        inter_route_2opt_star_first_improve,
        inter_route_cross_exchange_first_improve,
    ]

    current_vd = copy.deepcopy(vehicle_data)

    i = 0
    iterations = 0
    start_time = time.time()
    while i < len(neighborhoods) and iterations < iteration_max:

        operator = neighborhoods[i]
        improved = False

        # ------------------ 混合搜索部分 ----------------
        all_route_distance = calculate_distance_routes(current_vd, distance_matrix)
        current_distance = np.sum(all_route_distance)

        all_route_time_robustness, all_route_demand_robustness = cal_robustness_index_routes(
            current_vd, problem_data, travel_time_matrix, travel_time_disturbance, demand_disturbance)
        current_robustness_indicator = cal_robustness_index(
            all_route_time_robustness, all_route_demand_robustness)
        
        # 更新关键路径集
        key_routes_distance = update_key_routes_distance(
            current_vd, all_route_distance, p_kr
        )
        key_routes_robustness = update_key_routes_robustness(
            current_vd, all_route_time_robustness, all_route_demand_robustness, p_kr
        )
        # 两个关键路径集取并集
        key_routes = list(set(key_routes_distance) | set(key_routes_robustness))

        for search_index in key_routes:
            new_vd, new_obj, improved_flag, new_intermediate_variable = operator(
                current_vd,
                search_index,
                problem_data,
                distance_matrix,
                travel_time_matrix,
                (current_distance, current_robustness_indicator),
                all_route_distance,
                all_route_time_robustness,
                all_route_demand_robustness,
                travel_time_disturbance=travel_time_disturbance,
                demand_disturbance=demand_disturbance,
            )

            if not improved_flag:
                continue

            # 有改进，接收新解
            improved = True
            current_vd = new_vd
            current_distance, current_robustness_indicator = new_obj
            all_route_distance, all_route_time_robustness, all_route_demand_robustness = new_intermediate_variable

        # ---------------- 逻辑控制 ----------------

        iterations += 1
        if iterations > iteration_max:
            # print(f"VND run time exceeded timeout, terminating search")
            break
        # 检查是否有改进
        if not improved:
            # 如果没有改进，尝试下一个算子
            i += 1
            continue
        else:
            i = 0
    return current_vd


def _choose_better_candidate(first_candidate, second_candidate):
    """Choose the better improved candidate using the shared KRVND rule."""
    first_vd, first_obj, first_improved, first_mid = first_candidate
    second_vd, second_obj, second_improved, second_mid = second_candidate

    if not first_improved:
        return second_candidate
    if not second_improved:
        return first_candidate

    if accept_solution((first_vd, first_obj), (second_vd, second_obj)) == 2:
        return second_candidate
    return first_candidate


def relocate_first_improve(
    vehicle_data,
    search_index,
    problem_data,
    distance_matrix,
    travel_time_matrix,
    objective_value,
    all_route_distance,
    all_route_time_robustness,
    all_route_demand_robustness,
    travel_time_disturbance=0,
    demand_disturbance=0,
):
    """Relocate operator covering both intra-route and inter-route moves."""
    intra_candidate = intra_route_relocate_first_improve(
        vehicle_data,
        search_index,
        problem_data,
        distance_matrix,
        travel_time_matrix,
        objective_value,
        all_route_distance,
        all_route_time_robustness,
        all_route_demand_robustness,
        travel_time_disturbance=travel_time_disturbance,
        demand_disturbance=demand_disturbance,
    )
    inter_candidate = inter_route_relocate_first_improve(
        vehicle_data,
        search_index,
        problem_data,
        distance_matrix,
        travel_time_matrix,
        objective_value,
        all_route_distance,
        all_route_time_robustness,
        all_route_demand_robustness,
        travel_time_disturbance=travel_time_disturbance,
        demand_disturbance=demand_disturbance,
    )
    return _choose_better_candidate(intra_candidate, inter_candidate)


def swap_first_improve(
    vehicle_data,
    search_index,
    problem_data,
    distance_matrix,
    travel_time_matrix,
    objective_value,
    all_route_distance,
    all_route_time_robustness,
    all_route_demand_robustness,
    travel_time_disturbance=0,
    demand_disturbance=0,
):
    """Swap operator covering both intra-route and inter-route exchanges."""
    intra_candidate = intra_route_swap_first_improve(
        vehicle_data,
        search_index,
        problem_data,
        distance_matrix,
        travel_time_matrix,
        objective_value,
        all_route_distance,
        all_route_time_robustness,
        all_route_demand_robustness,
        travel_time_disturbance=travel_time_disturbance,
        demand_disturbance=demand_disturbance,
    )
    inter_candidate = inter_route_swap_first_improve(
        vehicle_data,
        search_index,
        problem_data,
        distance_matrix,
        travel_time_matrix,
        objective_value,
        all_route_distance,
        all_route_time_robustness,
        all_route_demand_robustness,
        travel_time_disturbance=travel_time_disturbance,
        demand_disturbance=demand_disturbance,
    )
    return _choose_better_candidate(intra_candidate, inter_candidate)

    
def intra_route_relocate_first_improve(
    vehicle_data,
    search_index,
    problem_data,
    distance_matrix,
    travel_time_matrix,
    objective_value,
    all_route_distance,
    all_route_time_robustness,
    all_route_demand_robustness,
    travel_time_disturbance=0,
    demand_disturbance=0,
):
    """intra-route relocate（first-improvement, balance-only）。"""
    if not (0 <= search_index < len(vehicle_data)):
        raise IndexError(f"insert_index={search_index} 超出 vehicle_data 范围")

    ori_vehicle = vehicle_data[search_index]
    route = ori_vehicle['unserved_customers']
    if len(route) <= 2:
        return vehicle_data, objective_value, False, None

    base_solution = (copy.deepcopy(vehicle_data), objective_value)

    for from_idx in range(1, len(route) - 1):
        customer = route[from_idx]
        reduced = route[:from_idx] + route[from_idx + 1:]
        for to_idx in range(1, len(reduced)):
            new_route = reduced[:to_idx] + [customer] + reduced[to_idx:]
            if new_route == route:
                continue

            trial_vd = copy.deepcopy(vehicle_data)
            trial_vehicle = trial_vd[search_index]
            trial_vehicle['unserved_customers'] = new_route

            trial_route_dist = calculate_distance_single_route(trial_vehicle, distance_matrix)
            trial_all_dist = all_route_distance.copy()
            trial_all_dist[search_index] = trial_route_dist
            trial_distance = sum(trial_all_dist)

            trial_t, trial_d = cal_robustness_index_single_route(
                trial_vehicle,
                problem_data,
                travel_time_matrix,
                travel_time_disturbance,
                demand_disturbance,
            )
            trial_all_t = all_route_time_robustness.copy()
            trial_all_d = all_route_demand_robustness.copy()
            trial_all_t[search_index] = trial_t
            trial_all_d[search_index] = trial_d
            trial_rob, feasible = cal_robustness_index(trial_all_t, trial_all_d, get_feasibility=True)
            if not feasible:
                continue

            trial_obj = (trial_distance, trial_rob)
            trial_solution = (trial_vd, trial_obj)
            if accept_solution(base_solution, trial_solution) == 2:
                return trial_vd, trial_obj, True, (trial_all_dist, trial_all_t, trial_all_d)

    return vehicle_data, objective_value, False, None


def intra_route_2opt_first_improve(
    vehicle_data,
    search_index,
    problem_data,
    distance_matrix,
    travel_time_matrix,
    objective_value,
    all_route_distance,
    all_route_time_robustness,
    all_route_demand_robustness,
    travel_time_disturbance=0,
    demand_disturbance=0,
):
    """intra-route 2-opt（first-improvement, balance-only）。"""
    if not (0 <= search_index < len(vehicle_data)):
        raise IndexError(f"search_index={search_index} 超出 vehicle_data 范围")

    route = vehicle_data[search_index]['unserved_customers']
    if len(route) <= 3:
        return vehicle_data, objective_value, False, None

    base_solution = (copy.deepcopy(vehicle_data), objective_value)
    n = len(route)
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
            if new_route == route:
                continue

            trial_vd = copy.deepcopy(vehicle_data)
            trial_vehicle = trial_vd[search_index]
            trial_vehicle['unserved_customers'] = new_route

            trial_route_dist = calculate_distance_single_route(trial_vehicle, distance_matrix)
            trial_all_dist = all_route_distance.copy()
            trial_all_dist[search_index] = trial_route_dist
            trial_distance = sum(trial_all_dist)

            trial_t, trial_d = cal_robustness_index_single_route(
                trial_vehicle,
                problem_data,
                travel_time_matrix,
                travel_time_disturbance,
                demand_disturbance,
            )
            trial_all_t = all_route_time_robustness.copy()
            trial_all_d = all_route_demand_robustness.copy()
            trial_all_t[search_index] = trial_t
            trial_all_d[search_index] = trial_d
            trial_rob, feasible = cal_robustness_index(trial_all_t, trial_all_d, get_feasibility=True)
            if not feasible:
                continue

            trial_obj = (trial_distance, trial_rob)
            trial_solution = (trial_vd, trial_obj)
            if accept_solution(base_solution, trial_solution) == 2:
                return trial_vd, trial_obj, True, (trial_all_dist, trial_all_t, trial_all_d)

    return vehicle_data, objective_value, False, None


def intra_route_oropt_first_improve(
    vehicle_data,
    search_index,
    problem_data,
    distance_matrix,
    travel_time_matrix,
    objective_value,
    all_route_distance,
    all_route_time_robustness,
    all_route_demand_robustness,
    travel_time_disturbance=0,
    demand_disturbance=0,
):
    """intra-route or-opt（first-improvement, balance-only）。"""
    if not (0 <= search_index < len(vehicle_data)):
        raise IndexError(f"search_index={search_index} 超出 vehicle_data 范围")

    route = vehicle_data[search_index]['unserved_customers']
    if len(route) <= 3:
        return vehicle_data, objective_value, False, None

    base_solution = (copy.deepcopy(vehicle_data), objective_value)
    n = len(route)
    for L in (2, 3):
        if n - 2 < L:
            continue
        for a in range(1, n - L):
            block = route[a:a + L]
            remain = route[:a] + route[a + L:]
            for b in range(1, len(remain)):
                if b == a:
                    continue
                new_route = remain[:b] + block + remain[b:]
                if new_route == route:
                    continue

                trial_vd = copy.deepcopy(vehicle_data)
                trial_vehicle = trial_vd[search_index]
                trial_vehicle['unserved_customers'] = new_route

                trial_route_dist = calculate_distance_single_route(trial_vehicle, distance_matrix)
                trial_all_dist = all_route_distance.copy()
                trial_all_dist[search_index] = trial_route_dist
                trial_distance = sum(trial_all_dist)

                trial_t, trial_d = cal_robustness_index_single_route(
                    trial_vehicle,
                    problem_data,
                    travel_time_matrix,
                    travel_time_disturbance,
                    demand_disturbance,
                )
                trial_all_t = all_route_time_robustness.copy()
                trial_all_d = all_route_demand_robustness.copy()
                trial_all_t[search_index] = trial_t
                trial_all_d[search_index] = trial_d
                trial_rob, feasible = cal_robustness_index(trial_all_t, trial_all_d, get_feasibility=True)
                if not feasible:
                    continue

                trial_obj = (trial_distance, trial_rob)
                trial_solution = (trial_vd, trial_obj)
                if accept_solution(base_solution, trial_solution) == 2:
                    return trial_vd, trial_obj, True, (trial_all_dist, trial_all_t, trial_all_d)

    return vehicle_data, objective_value, False, None


def intra_route_swap_first_improve(
    vehicle_data,
    search_index,
    problem_data,
    distance_matrix,
    travel_time_matrix,
    objective_value,
    all_route_distance,
    all_route_time_robustness,
    all_route_demand_robustness,
    travel_time_disturbance=0,
    demand_disturbance=0,
):
    """intra-route swap（first-improvement, balance-only）。"""
    if not (0 <= search_index < len(vehicle_data)):
        raise IndexError(f"search_index={search_index} 超出 vehicle_data 范围")

    route = vehicle_data[search_index]['unserved_customers']
    if len(route) <= 3:
        return vehicle_data, objective_value, False, None

    base_solution = (copy.deepcopy(vehicle_data), objective_value)
    n = len(route)
    for i in range(1, n - 1):
        for j in range(i + 1, n - 1):
            new_route = route.copy()
            new_route[i], new_route[j] = new_route[j], new_route[i]

            trial_vd = copy.deepcopy(vehicle_data)
            trial_vehicle = trial_vd[search_index]
            trial_vehicle['unserved_customers'] = new_route

            trial_route_dist = calculate_distance_single_route(trial_vehicle, distance_matrix)
            trial_all_dist = all_route_distance.copy()
            trial_all_dist[search_index] = trial_route_dist
            trial_distance = sum(trial_all_dist)

            trial_t, trial_d = cal_robustness_index_single_route(
                trial_vehicle,
                problem_data,
                travel_time_matrix,
                travel_time_disturbance,
                demand_disturbance,
            )
            trial_all_t = all_route_time_robustness.copy()
            trial_all_d = all_route_demand_robustness.copy()
            trial_all_t[search_index] = trial_t
            trial_all_d[search_index] = trial_d
            trial_rob, feasible = cal_robustness_index(trial_all_t, trial_all_d, get_feasibility=True)
            if not feasible:
                continue

            trial_obj = (trial_distance, trial_rob)
            trial_solution = (trial_vd, trial_obj)
            if accept_solution(base_solution, trial_solution) == 2:
                return trial_vd, trial_obj, True, (trial_all_dist, trial_all_t, trial_all_d)

    return vehicle_data, objective_value, False, None


def inter_route_relocate_first_improve(
    vehicle_data,
    search_index,
    problem_data,
    distance_matrix,
    travel_time_matrix,
    objective_value,
    all_route_distance,
    all_route_time_robustness,
    all_route_demand_robustness,
    travel_time_disturbance=0,
    demand_disturbance=0,
):
    """inter-route relocate（first-improvement, balance-only）。"""
    if not (0 <= search_index < len(vehicle_data)):
        raise IndexError(f"search_index={search_index} 超出 vehicle_data 范围")

    origin_route = vehicle_data[search_index]['unserved_customers']
    if len(origin_route) <= 2:
        return vehicle_data, objective_value, False, None

    base_solution = (copy.deepcopy(vehicle_data), objective_value)
    num_vehicles = len(vehicle_data)

    for pos in range(1, len(origin_route) - 1):
        cust = origin_route[pos]
        for dest_index in range(num_vehicles):
            if dest_index == search_index:
                continue
            dest_route = vehicle_data[dest_index].get('unserved_customers')
            if not dest_route or len(dest_route) <= 1:
                continue
            for b in range(1, len(dest_route)):
                trial_vd = copy.deepcopy(vehicle_data)
                trial_ori = trial_vd[search_index]
                trial_dest = trial_vd[dest_index]
                trial_ori['unserved_customers'].remove(cust)
                trial_dest['unserved_customers'].insert(b, cust)

                trial_all = all_route_distance.copy()
                trial_all[search_index] = calculate_distance_single_route(trial_ori, distance_matrix)
                trial_all[dest_index] = calculate_distance_single_route(trial_dest, distance_matrix)
                trial_distance = sum(trial_all)

                ori_t, ori_d = cal_robustness_index_single_route(
                    trial_ori, problem_data, travel_time_matrix, travel_time_disturbance, demand_disturbance
                )
                dst_t, dst_d = cal_robustness_index_single_route(
                    trial_dest, problem_data, travel_time_matrix, travel_time_disturbance, demand_disturbance
                )
                trial_times = all_route_time_robustness.copy()
                trial_demands = all_route_demand_robustness.copy()
                trial_times[search_index] = ori_t
                trial_demands[search_index] = ori_d
                trial_times[dest_index] = dst_t
                trial_demands[dest_index] = dst_d
                trial_rob, feasible = cal_robustness_index(trial_times, trial_demands, get_feasibility=True)
                if not feasible:
                    continue

                trial_obj = (trial_distance, trial_rob)
                trial_solution = (trial_vd, trial_obj)
                if accept_solution(base_solution, trial_solution) == 2:
                    return trial_vd, trial_obj, True, (trial_all, trial_times, trial_demands)

    return vehicle_data, objective_value, False, None


def inter_route_swap_first_improve(
    vehicle_data,
    search_index,
    problem_data,
    distance_matrix,
    travel_time_matrix,
    objective_value,
    all_route_distance,
    all_route_time_robustness,
    all_route_demand_robustness,
    travel_time_disturbance=0,
    demand_disturbance=0,
):
    """inter-route swap（first-improvement, balance-only）。"""
    num_vehicles = len(vehicle_data)
    if not (0 <= search_index < num_vehicles):
        raise IndexError(f"search_index={search_index} 超出 vehicle_data 范围")

    origin_route = vehicle_data[search_index]['unserved_customers']
    if len(origin_route) <= 2:
        return vehicle_data, objective_value, False, None

    base_solution = (copy.deepcopy(vehicle_data), objective_value)

    for pos_i in range(1, len(origin_route) - 1):
        cust_i = origin_route[pos_i]
        for dest_index in range(num_vehicles):
            if dest_index == search_index:
                continue
            dest_route = vehicle_data[dest_index].get('unserved_customers')
            if not dest_route or len(dest_route) <= 2:
                continue
            for pos_j in range(1, len(dest_route) - 1):
                cust_j = dest_route[pos_j]

                trial_vd = copy.deepcopy(vehicle_data)
                trial_ori = trial_vd[search_index]
                trial_dest = trial_vd[dest_index]
                idx_i = trial_ori['unserved_customers'].index(cust_i)
                idx_j = trial_dest['unserved_customers'].index(cust_j)
                trial_ori['unserved_customers'][idx_i], trial_dest['unserved_customers'][idx_j] = (
                    trial_dest['unserved_customers'][idx_j],
                    trial_ori['unserved_customers'][idx_i],
                )

                trial_all = all_route_distance.copy()
                trial_all[search_index] = calculate_distance_single_route(trial_ori, distance_matrix)
                trial_all[dest_index] = calculate_distance_single_route(trial_dest, distance_matrix)
                trial_distance = sum(trial_all)

                ori_t, ori_d = cal_robustness_index_single_route(
                    trial_ori, problem_data, travel_time_matrix, travel_time_disturbance, demand_disturbance
                )
                dst_t, dst_d = cal_robustness_index_single_route(
                    trial_dest, problem_data, travel_time_matrix, travel_time_disturbance, demand_disturbance
                )
                trial_times = all_route_time_robustness.copy()
                trial_demands = all_route_demand_robustness.copy()
                trial_times[search_index] = ori_t
                trial_demands[search_index] = ori_d
                trial_times[dest_index] = dst_t
                trial_demands[dest_index] = dst_d
                trial_rob, feasible = cal_robustness_index(trial_times, trial_demands, get_feasibility=True)
                if not feasible:
                    continue

                trial_obj = (trial_distance, trial_rob)
                trial_solution = (trial_vd, trial_obj)
                if accept_solution(base_solution, trial_solution) == 2:
                    return trial_vd, trial_obj, True, (trial_all, trial_times, trial_demands)

    return vehicle_data, objective_value, False, None


def inter_route_2opt_star_first_improve(
    vehicle_data,
    search_index,
    problem_data,
    distance_matrix,
    travel_time_matrix,
    objective_value,
    all_route_distance,
    all_route_time_robustness,
    all_route_demand_robustness,
    travel_time_disturbance=0,
    demand_disturbance=0,
):
    """inter-route 2-opt*（first-improvement, balance-only）。"""
    num_vehicles = len(vehicle_data)
    if not (0 <= search_index < num_vehicles):
        raise IndexError(f"search_index={search_index} 超出 vehicle_data 范围")

    origin_route = vehicle_data[search_index]['unserved_customers']
    if len(origin_route) <= 2:
        return vehicle_data, objective_value, False, None

    base_solution = (copy.deepcopy(vehicle_data), objective_value)

    for dest_index in range(num_vehicles):
        if dest_index == search_index:
            continue
        dest_route = vehicle_data[dest_index].get('unserved_customers')
        if not dest_route or len(dest_route) <= 2:
            continue
        for pos_i in range(1, len(origin_route) - 1):
            for pos_j in range(1, len(dest_route) - 1):
                trial_vd = copy.deepcopy(vehicle_data)
                trial_ori = trial_vd[search_index]
                trial_dest = trial_vd[dest_index]
                ori_r = trial_ori['unserved_customers']
                dst_r = trial_dest['unserved_customers']

                new_r1 = ori_r[:pos_i] + dst_r[pos_j:]
                new_r2 = dst_r[:pos_j] + ori_r[pos_i:]
                if new_r1 == ori_r and new_r2 == dst_r:
                    continue
                trial_ori['unserved_customers'] = new_r1
                trial_dest['unserved_customers'] = new_r2

                trial_all = all_route_distance.copy()
                trial_all[search_index] = calculate_distance_single_route(trial_ori, distance_matrix)
                trial_all[dest_index] = calculate_distance_single_route(trial_dest, distance_matrix)
                trial_distance = sum(trial_all)

                ori_t, ori_d = cal_robustness_index_single_route(
                    trial_ori, problem_data, travel_time_matrix, travel_time_disturbance, demand_disturbance
                )
                dst_t, dst_d = cal_robustness_index_single_route(
                    trial_dest, problem_data, travel_time_matrix, travel_time_disturbance, demand_disturbance
                )
                trial_times = all_route_time_robustness.copy()
                trial_demands = all_route_demand_robustness.copy()
                trial_times[search_index] = ori_t
                trial_demands[search_index] = ori_d
                trial_times[dest_index] = dst_t
                trial_demands[dest_index] = dst_d
                trial_rob, feasible = cal_robustness_index(trial_times, trial_demands, get_feasibility=True)
                if not feasible:
                    continue

                trial_obj = (trial_distance, trial_rob)
                trial_solution = (trial_vd, trial_obj)
                if accept_solution(base_solution, trial_solution) == 2:
                    return trial_vd, trial_obj, True, (trial_all, trial_times, trial_demands)

    return vehicle_data, objective_value, False, None


def inter_route_cross_exchange_first_improve(
    vehicle_data,
    search_index,
    problem_data,
    distance_matrix,
    travel_time_matrix,
    objective_value,
    all_route_distance,
    all_route_time_robustness,
    all_route_demand_robustness,
    travel_time_disturbance=0,
    demand_disturbance=0,
):
    """inter-route cross-exchange（first-improvement, balance-only）。"""
    num_vehicles = len(vehicle_data)
    if not (0 <= search_index < num_vehicles):
        raise IndexError(f"search_index={search_index} 超出 vehicle_data 范围")

    origin_route = vehicle_data[search_index]['unserved_customers']
    if len(origin_route) <= 3:
        return vehicle_data, objective_value, False, None

    base_solution = (copy.deepcopy(vehicle_data), objective_value)

    for dest_index in range(num_vehicles):
        if dest_index == search_index:
            continue
        dest_route = vehicle_data[dest_index].get('unserved_customers')
        if not dest_route or len(dest_route) <= 3:
            continue

        for i_start in range(1, len(origin_route) - 1):
            for i_end in range(i_start, len(origin_route) - 1):
                for j_start in range(1, len(dest_route) - 1):
                    for j_end in range(j_start, len(dest_route) - 1):
                        trial_vd = copy.deepcopy(vehicle_data)
                        trial_ori = trial_vd[search_index]
                        trial_dest = trial_vd[dest_index]
                        ori_r = trial_ori['unserved_customers']
                        dst_r = trial_dest['unserved_customers']

                        seg1 = ori_r[i_start:i_end + 1]
                        seg2 = dst_r[j_start:j_end + 1]
                        new_r1 = ori_r[:i_start] + seg2 + ori_r[i_end + 1:]
                        new_r2 = dst_r[:j_start] + seg1 + dst_r[j_end + 1:]
                        if new_r1 == ori_r and new_r2 == dst_r:
                            continue
                        trial_ori['unserved_customers'] = new_r1
                        trial_dest['unserved_customers'] = new_r2

                        trial_all = all_route_distance.copy()
                        trial_all[search_index] = calculate_distance_single_route(trial_ori, distance_matrix)
                        trial_all[dest_index] = calculate_distance_single_route(trial_dest, distance_matrix)
                        trial_distance = sum(trial_all)

                        ori_t, ori_d = cal_robustness_index_single_route(
                            trial_ori, problem_data, travel_time_matrix, travel_time_disturbance, demand_disturbance
                        )
                        dst_t, dst_d = cal_robustness_index_single_route(
                            trial_dest, problem_data, travel_time_matrix, travel_time_disturbance, demand_disturbance
                        )
                        trial_times = all_route_time_robustness.copy()
                        trial_demands = all_route_demand_robustness.copy()
                        trial_times[search_index] = ori_t
                        trial_demands[search_index] = ori_d
                        trial_times[dest_index] = dst_t
                        trial_demands[dest_index] = dst_d
                        trial_rob, feasible = cal_robustness_index(trial_times, trial_demands, get_feasibility=True)
                        if not feasible:
                            continue

                        trial_obj = (trial_distance, trial_rob)
                        trial_solution = (trial_vd, trial_obj)
                        if accept_solution(base_solution, trial_solution) == 2:
                            return trial_vd, trial_obj, True, (trial_all, trial_times, trial_demands)

    return vehicle_data, objective_value, False, None


def calculate_distance_single_route(vehicle, distance_matrix):
    '''
    计算单条路径的距离
    :param vehicle: 车辆数据
    :param distance_matrix: 距离矩阵
    :return: 目标函数值
    '''

    distance = 0

    if vehicle['unserved_customers']:
        route = vehicle['unserved_customers']
        for i in range(1, len(route)):
            customer = route[i]
            last_customer = route[i - 1]
            # 计算路径成本
            distance += distance_matrix[last_customer, customer]

    return distance

def calculate_distance_routes(vehicle_data, distance_matrix):
    '''
    计算整个路径方案的目标函数中间值
    :param vehicle_data: 所有车辆数据
    :param distance_matrix: 距离矩阵
    :return: 目标函数值
    '''
    all_route_distances = []
    for vehicle in vehicle_data:
        if len(vehicle['unserved_customers']) <= 1:
            route_distance = 0
        else:
            route_distance = calculate_distance_single_route(vehicle, distance_matrix)
        all_route_distances.append(route_distance)
    return all_route_distances

def update_distance(vehicle_list, all_route_distances,
                            problem_data, distance_matrix):
    '''
    当只有单个或多个路径改变时，更新整个路径方案的路径距离
    :param vehicle: 当前车辆数据
    :param all_route_distances: 所有路径的距离列表
    :param problem_data: 问题数据
    :param distance_matrix: 距离矩阵
    :return: 更新后的目标函数值
    '''
    all_route_distances = all_route_distances.copy()
    # 计算单条路径的目标函数中间值
    for vehicle in vehicle_list:
        distance = calculate_distance_single_route(vehicle, distance_matrix)
        update_index = vehicle['vehicle_id']
        # 更新对应路径的目标函数值
        all_route_distances[update_index] = distance
    # 计算总目标函数值
    distance = sum(all_route_distances)
    return distance

def cal_robustness_index_routes(vehicle_data, problem_data, travel_time_matrix, travel_time_disturbance=0, demand_disturbance=0):
    '''
    计算路径方案的鲁棒性指标
    :param vehicle_data: 车辆数据
    :param problem_data: 客户数据
    :param travel_time_matrix: 旅行时间矩阵
    :param travel_time_disturbance: 旅行时间扰动
    :param demand_disturbance: 需求扰动
    :return: 每条路径包含的每个客户的时间鲁棒性指标和需求鲁棒性指标
    '''
    robustness_time_routes = []
    robustness_demand_routes = []
    for vehicle in vehicle_data:
        route = vehicle['unserved_customers']
        if len(route) <= 1:
            # 如果路径中没有中间客户，则不计算当前路径的鲁棒性指标
            robustness_time_routes.append([])
            robustness_demand_routes.append(None)
            continue

        # 计算每条路径的鲁棒性指标
        leave_last_customer_time = vehicle['expected_service_end_time']
        current_time = leave_last_customer_time
        demand = vehicle['used_capacity']
        if route[0] != 0:
            demand += problem_data['DEMAND'][route[0]]
        robustness_time_customers = []

        for i in range(1, len(route)):

            customer = route[i]
            previous_customer = route[i - 1]

            travel_time = travel_time_matrix[previous_customer][customer]
            arrive_time = current_time + travel_time + travel_time_disturbance
            ready_time = problem_data['READY_TIME'][customer]
            due_date = problem_data['DUE_DATE'][customer]
            service_time = problem_data['SERVICE_TIME'][customer]

            # 论文中的 Rt_i 只针对客户节点计算，不统计 depot
            if customer != 0:
                if travel_time_disturbance == 0:
                    robustness_time = 1.0
                else:
                    robustness_time = max(min((due_date - arrive_time) / travel_time_disturbance, 1), 0)
                robustness_time_customers.append(robustness_time)

            current_time = max(arrive_time, ready_time) + service_time
            if customer != 0:
                demand += problem_data['DEMAND'][customer]

        if demand_disturbance == 0:
            robustness_demand = 1.0
        else:
            customer_count = sum(1 for customer in route if customer != 0)
            if customer_count == 0:
                robustness_demand = 1.0
            else:
                robustness_demand = max(min((problem_data['capacity'] - demand) / (customer_count * demand_disturbance), 1), 0)
        # 时间鲁棒性指标直接保存每个客户的指标
        robustness_time_routes.append(robustness_time_customers)
        robustness_demand_routes.append(robustness_demand)

    return robustness_time_routes, robustness_demand_routes

def cal_robustness_index_single_route(vehicle, problem_data, travel_time_matrix, travel_time_disturbance=0, demand_disturbance=0):
    '''
    计算单条路径的鲁棒性指标
    :param vehicle: 车辆数据
    :param problem_data: 客户数据
    :param travel_time_matrix: 旅行时间矩阵
    :param travel_time_disturbance: 旅行时间扰动
    :param demand_disturbance: 需求扰动
    :return: 每条路径的时间鲁棒性指标和需求鲁棒性指标
    '''


    route = vehicle['unserved_customers']
    if len(route) <= 1:
        # 如果路径中没有中间客户，则不计算当前路径的鲁棒性指标
        return [], None
    # 计算每条路径的鲁棒性指标
    leave_last_customer_time = vehicle['expected_service_end_time']
    current_time = leave_last_customer_time
    demand = vehicle['used_capacity']
    if route[0] != 0:
        demand += problem_data['DEMAND'][route[0]]
    robustness_time_customers = []

    for i in range(1, len(route)):

        customer = route[i]
        previous_customer = route[i - 1]

        travel_time = travel_time_matrix[previous_customer][customer]
        arrive_time = current_time + travel_time + travel_time_disturbance
        ready_time = problem_data['READY_TIME'][customer]
        due_date = problem_data['DUE_DATE'][customer]
        service_time = problem_data['SERVICE_TIME'][customer]

        if customer != 0:
            if travel_time_disturbance == 0:
                robustness_time = 1.0
            else:
                robustness_time = max(min((due_date - arrive_time) / travel_time_disturbance, 1), 0)
            robustness_time_customers.append(robustness_time)

        current_time = max(arrive_time, ready_time) + service_time
        if customer != 0:
            demand += problem_data['DEMAND'][customer]
    if demand_disturbance == 0:
        robustness_demand_route = 1.0
    else:
        customer_count = sum(1 for customer in route if customer != 0)
        if customer_count == 0:
            robustness_demand_route = 1.0
        else:
            robustness_demand_route = max(min((problem_data['capacity'] - demand) / (customer_count * demand_disturbance), 1), 0)
    # 时间鲁棒性指标直接保存每个客户的指标
    robustness_time_route = robustness_time_customers

    return robustness_time_route, robustness_demand_route

def cal_robustness_index(robustness_time_routes, robustness_demand_routes, get_feasibility=False):
    """
    计算路径方案总体的鲁棒性指标
    :param robustness_time_routes: 每条路径的时间鲁棒性指标
    :param robustness_demand_routes: 每条路径的需求鲁棒性指标
    :return: 总体的时间鲁棒性指标和需求鲁棒性指标
    """
    # 展平时间鲁棒性指标
    all_robustness_time = [item for sublist in robustness_time_routes for item in sublist]
    # 计算调和平均时间鲁棒性指标
    if len(all_robustness_time) == 0:
        final_robustness_time = 0
    else:
        # 如果有任何一个鲁棒性时间为0，则最终鲁棒性时间为0
        if 0 in all_robustness_time:
            final_robustness_time = 0
        else:
            # 计算调和平均
            final_robustness_time = len(all_robustness_time) / np.sum(1 / np.array(all_robustness_time))

    robustness_demand_routes = [demand for demand in robustness_demand_routes if demand is not None]
    # 计算调和平均需求鲁棒性指标
    if len(robustness_demand_routes) == 0:
        final_robustness_demand = 0
    else:
        # 如果有任何一个鲁棒性需求为0，则最终鲁棒性需求为0
        if 0 in robustness_demand_routes:
            final_robustness_demand = 0
        else:
            # 计算调和平均
            final_robustness_demand = len(robustness_demand_routes) / np.sum(1 / np.array(robustness_demand_routes))

    final_robustness = (final_robustness_time + final_robustness_demand) / 2

    # 返回最终的鲁棒性指标
    if get_feasibility:
        # 如果需要返回可行性，则判断两个鲁棒性指标有没有为0的
        if final_robustness_time == 0 or final_robustness_demand == 0:
            return final_robustness, False
        else:
            # 否则返回True和总的鲁棒性指标
            return final_robustness, True
    else:
        # 否则只返回总的鲁棒性指标
        return final_robustness

def cal_robustness_index_devide(robustness_time_routes, robustness_demand_routes, get_feasibility=False):
    """
    计算路径方案总体的鲁棒性指标
    :param robustness_time_routes: 每条路径的时间鲁棒性指标
    :param robustness_demand_routes: 每条路径的需求鲁棒性指标
    :return: 总体的时间鲁棒性指标和需求鲁棒性指标
    """
    # 展平时间鲁棒性指标
    all_robustness_time = [item for sublist in robustness_time_routes for item in sublist]
    # 计算调和平均时间鲁棒性指标
    if len(all_robustness_time) == 0:
        final_robustness_time = 0
    else:
        # 如果有任何一个鲁棒性时间为0，则最终鲁棒性时间为0
        if 0 in all_robustness_time:
            final_robustness_time = 0
        else:
            # 计算调和平均
            final_robustness_time = len(all_robustness_time) / np.sum(1 / np.array(all_robustness_time))

    robustness_demand_routes = [demand for demand in robustness_demand_routes if demand is not None]
    # 计算调和平均需求鲁棒性指标
    if len(robustness_demand_routes) == 0:
        final_robustness_demand = 0
    else:
        # 如果有任何一个鲁棒性需求为0，则最终鲁棒性需求为0
        if 0 in robustness_demand_routes:
            final_robustness_demand = 0
        else:
            # 计算调和平均
            final_robustness_demand = len(robustness_demand_routes) / np.sum(1 / np.array(robustness_demand_routes))

    # 返回最终的鲁棒性指标
    if get_feasibility:
        # 如果需要返回可行性，则判断两个鲁棒性指标有没有为0的
        if final_robustness_time == 0 or final_robustness_demand == 0:
            return final_robustness_time, final_robustness_demand, False
        else:
            # 否则返回True和总的鲁棒性指标
            return final_robustness_time, final_robustness_demand, True
    else:
        # 否则只返回总的鲁棒性指标
        return final_robustness_time, final_robustness_demand


def check_feasibility(vehicle, problem_data, travel_time_matrix, delay_tolerance_time=0):
    '''
    检查车辆路径的可行性
    :param vehicle_data: 车辆数据
    :param problem_data: 问题数据
    :param travel_time_matrix: 旅行时间矩阵
    :param objective_index: 目标函数索引
    :param delay_tolerance_time: 延误容忍时间
    :return: 是否可行, 约束违反量
    '''
    capacity = problem_data['capacity']
    capacity_feasible = True
    time_window_feasible = True

    capacity_used = 0
    if vehicle['served_customers']:
        # 检查车辆容量是否满足
        capacity_used = sum(problem_data['DEMAND'][customer] for customer in vehicle['served_customers'])

    if vehicle['unserved_customers']:
        route = vehicle['unserved_customers']
        leave_last_customer_time = vehicle['expected_service_end_time']
        time = leave_last_customer_time
        delay_time = 0
        last_customer = route[0]  # 第一个客户是仓库
        for customer in route[1:]:  # 跳过第一个客户（仓库）
            # 计算到达当前客户的时间
            arrive_time = max(time + travel_time_matrix[last_customer, customer], problem_data['READY_TIME'][customer])
            # 计算延误时间
            delay_time_customer = max(0, arrive_time - problem_data['DUE_DATE'][customer])
            if delay_time_customer > delay_tolerance_time: # 如果延误时间超过容忍度，则不满足时间窗要求
                # print(f"车辆 {vehicle['vehicle_id']} 在客户 {customer} 上存在延误时间: {delay_time}")
                time_window_feasible = False
            if customer == 0:
                if delay_time_customer > 0: # 如果车辆返回仓库时存在延误时间，即直接不满足时间窗要求
                    # print(f"车辆 {vehicle['vehicle_id']} 返回仓库时存在延误时间: {delay_time}")
                    time_window_feasible = False
            delay_time += delay_time_customer
            # 更新时间  
            time = arrive_time + problem_data['SERVICE_TIME'][customer]

            capacity_used += problem_data['DEMAND'][customer]

            last_customer = customer

    if capacity_used > capacity:
        # print(f"车辆 {vehicle['vehicle_id']} 的容量超出限制: {capacity_used} > {capacity}")
        capacity_feasible = False
        capacity_violate = capacity_used - capacity
    else:
        capacity_violate = 0

    return capacity_feasible and time_window_feasible, capacity_violate + delay_time

def is_dominated(obj1, obj2):
    """
    检查 objective1 是否支配 objective2。
    :param objective1: 第一个目标
    :param objective2: 第二个目标
    :return: True if solution1 dominates solution2, False otherwise
    """
    if len(obj1) == 1:
        # 如果只有一个目标，直接比较大小
        return obj1[0] < obj2[0]
    # 检查是否在所有目标上都不劣于 solution2，且至少在一个目标上优于 solution2
    is_dominated = all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 < o2 for o1, o2 in zip(obj1, obj2))
    
    return is_dominated


def accept_solution(solution1, solution2):
    """
    仅按 balance 目标比较两个解。
    :param solution1: (路径方案, (distance, robustness))
    :param solution2: (路径方案, (distance, robustness))
    :return: 1 表示 solution1 更优, 2 表示 solution2 更优
    """
    _, obj1 = solution1
    _, obj2 = solution2

    distance1, robustness1 = obj1
    distance2, robustness2 = obj2

    if robustness1 > robustness2:
        return 1
    if robustness1 < robustness2:
        return 2
    if distance1 < distance2:
        return 1
    if distance1 > distance2:
        return 2
    return 1

def update_key_routes_distance(
    vehicle_data, all_route_distance, p_d
):
    """
    更新所有路径的距离列表。
    :param vehicle_data: 车辆数据
    :param all_route_distance: 所有路径的距离列表
    :param p_d: 选择单位距离最大的前p_d%个路径
    :return: 更新后的目标函数值
    """
    all_route_unit_distance = []
    for i in range(len(vehicle_data)):
        if len(vehicle_data[i]['unserved_customers']) <= 2:
            unit_distance = 0
        else:
            unit_distance = all_route_distance[i] / len(vehicle_data[i]['unserved_customers']) if vehicle_data[i]['unserved_customers'] else 0

        all_route_unit_distance.append(unit_distance)

    # 计算单位距离最大的前 p_d% 的路径索引
    num_key_routes = max(1, int(len(vehicle_data) * p_d))
    key_routes_distance = np.argsort(all_route_unit_distance)[-num_key_routes:]

    return key_routes_distance

def update_key_routes_robustness(
    vehicle_data, all_route_time_robustness, all_route_demand_robustness, p_r
):
    """
    更新所有路径的鲁棒性指标列表。
    :param vehicle_data: 车辆数据
    :param all_route_time_robustness: 所有路径的时间鲁棒性指标列表
    :param all_route_demand_robustness: 所有路径的需求鲁棒性指标列表
    :param p_r: 选择鲁棒性指标最小的前p_r%个路径
    :return: 更新后的目标函数值
    """
    all_route_robustness_index = []

    for i in range(len(all_route_time_robustness)):

        if len(vehicle_data[i]['unserved_customers']) <= 2:
            robustness_index = 10
        else:
            route_time_robustness = all_route_time_robustness[i]
            if 0 in route_time_robustness:
                robustness_time = 0
            else:
                # 计算调和平均
                robustness_time = len(route_time_robustness) / np.sum(1 / np.array(route_time_robustness))

            robustness_demand = all_route_demand_robustness[i]
            robustness_index = (robustness_time + robustness_demand) / 2

        all_route_robustness_index.append(robustness_index)

    # 计算前 p_r% 鲁棒性最小的路径索引
    num_key_routes = max(1, int(len(vehicle_data) * p_r))
    key_routes_robustness = np.argsort(all_route_robustness_index)[:num_key_routes]

    return key_routes_robustness
