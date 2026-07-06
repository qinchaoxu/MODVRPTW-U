'''
Description: 特定的局部搜索操作(如车辆数量、总成本、总等待时间、总延迟时间、高冗余空间均匀度指标等)的实现
'''

import numpy as np
import random
import copy
from typing import List, Dict, Any
from algorithm.robustness_calculation import cal_robustness_index_routes, cal_robustness_index_single_route, cal_robustness_index



def robustness_focused_local_search(solution: List[List[int]], data: Dict[str, Any], distance_matrix: np.ndarray, travel_time_matrix: np.ndarray, travel_time_disturbance: float, demand_disturbance: float) -> List[List[int]]:
    '''
    针对鲁棒性目标的局部搜索
    :param solution: 当前解
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
    :param travel_time_matrix: 旅行时间矩阵
    :param travel_time_disturbance: 旅行时间扰动
    :param demand_disturbance: 需求扰动
    :return: 更新后的解
    '''

    solution = robustness_focused_local_search_relocate(solution, data, distance_matrix, travel_time_matrix, travel_time_disturbance, demand_disturbance)
    solution = robustness_focused_local_search_2opt(solution, data, distance_matrix, travel_time_matrix, travel_time_disturbance, demand_disturbance)
    solution = robustness_focused_local_search_oropt(solution, data, distance_matrix, travel_time_matrix, travel_time_disturbance, demand_disturbance)
    solution = robustness_focused_local_search_exchange(solution, data, distance_matrix, travel_time_matrix, travel_time_disturbance, demand_disturbance)
    solution = robustness_focused_local_search_2opt_star(solution, data, distance_matrix, travel_time_matrix, travel_time_disturbance, demand_disturbance)
    solution = robustness_focused_local_search_cross_exchange(solution, data, distance_matrix, travel_time_matrix, travel_time_disturbance, demand_disturbance)


    # 以下实现随机选择
    # inter_route_ls_operator = random.choice([robustness_focused_local_search_relocate,
    #                                         robustness_focused_local_search_exchange,
    #                                         robustness_focused_local_search_2opt_star,
    #                                         robustness_focused_local_search_cross_exchange])
    
    # intra_route_ls_operator = random.choice([robustness_focused_local_search_2opt
    #                               , robustness_focused_local_search_oropt])
    
    # solution = intra_route_ls_operator(solution, data, distance_matrix, travel_time_matrix, travel_time_disturbance, demand_disturbance)

    # solution = inter_route_ls_operator(solution, data, distance_matrix, travel_time_matrix, travel_time_disturbance, demand_disturbance)
    return solution


def robustness_focused_local_search_relocate(solution: List[List[int]], data: Dict[str, Any], distance_matrix: np.ndarray, travel_time_matrix: np.ndarray, travel_time_disturbance: float, demand_disturbance: float) -> List[List[int]]:
    '''
    针对一般目标的relocate局部搜索操作
    :param solution: 当前解
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
    :param travel_time_matrix: 旅行时间矩阵
    :param travel_time_disturbance: 旅行时间扰动
    :param demand_disturbance: 需求扰动
    :return: 更新后的解
    '''
    routes = copy.deepcopy(solution)

    # 计算所有路径的鲁棒性指标
    robustness_time_routes, robustness_demand_routes = cal_robustness_index_routes(routes, data, travel_time_matrix, travel_time_disturbance, demand_disturbance)
    final_robustness_time_routes = []
    for route in robustness_time_routes:
        if len(route) == 0:
            final_robustness_time_routes.append(0)
        else:
            # 如果有任何一个客户的时间鲁棒性为0，则该路径的时间鲁棒性为0
            if any(time == 0 for time in route):
                final_robustness_time_routes.append(0)
            else:
                # 计算调和平均时间鲁棒性指标
                final_robustness_time_routes.append(len(route) / np.sum(1 / np.array(route)))
    robustness_overall_routes = [time + demand for time, demand in zip(final_robustness_time_routes, robustness_demand_routes)]

    # 选择鲁棒性指标最低的路径
    route_index_chosen = np.argmin(robustness_overall_routes)
    route_chosen = routes[route_index_chosen]

    # 随机抽取一个或多个客户
    if len(route_chosen) >= 4 and random.random() >= 0.5:
        k = random.randint(2, len(route_chosen)-2)
    else:
        k = 1
    customer_index_chosen = sorted(random.sample(range(1, len(route_chosen)-1), k=k))

    # 移除选中的客户
    customers_chosen = [route_chosen[i] for i in customer_index_chosen]
    for i in sorted(customer_index_chosen, reverse=True):
        del route_chosen[i]

    # 如果路径移除客户后为空，则删除该路径
    if len(route_chosen) <= 2:
        routes.remove(route_chosen)
        # 从鲁棒性指标列表中移除对应路径
        # robustness_overall_routes.pop(route_index_chosen)
        robustness_time_routes.pop(route_index_chosen)
        robustness_demand_routes.pop(route_index_chosen)
    else:
        # 如果路径仍然存在，则更新鲁棒性指标
        robustness_time_route, robustness_demand_route = cal_robustness_index_single_route(route_chosen, data, travel_time_matrix, travel_time_disturbance, demand_disturbance)
        # robustness_overall_route = robustness_time_route + robustness_demand_route
        # robustness_overall_routes[route_index_chosen] = robustness_overall_route
        robustness_time_routes[route_index_chosen] = robustness_time_route
        robustness_demand_routes[route_index_chosen] = robustness_demand_route

    customer_insert_fail = []

    # 插入到所有路径中的最佳位置（根据选择的目标函数）
    for customer in customers_chosen:
        best_route_index = -1
        best_route = None
        current_robustness_indicator = cal_robustness_index(robustness_time_routes, robustness_demand_routes)
        max_robustness_indicator = float('-inf')
        inserted = False
        accepted_move = False

        # 遍历所有路径，寻找最佳插入位置
        for i, route in enumerate(routes):
            
            # robustness_overall_routes_copy = robustness_overall_routes.copy()  # 复制鲁棒性指标列表，避免修改原列表
            robustness_time_routes_copy = robustness_time_routes.copy()  # 复制时间鲁棒性指标列表
            robustness_demand_routes_copy = robustness_demand_routes.copy()  # 复制需求鲁棒性指标列表

            # 尝试将客户插入到路径中的每个位置
            for j in range(1, len(route)):
                route_copy = route.copy()
                new_route = route_copy[:j] + [customer] + route_copy[j:]  # 在路径中插入客户
                # 检查路径可行性
                if check_route_feasible(new_route, data, travel_time_matrix):
                    # 计算插入后的鲁棒性指标
                    new_robustness_time, new_robustness_demand = cal_robustness_index_single_route(new_route, data, travel_time_matrix, travel_time_disturbance, demand_disturbance)
                    # new_robustness_overall = new_robustness_time + new_robustness_demand

                    # 更新robustness_overall_routes
                    robustness_time_routes_copy[i] = new_robustness_time
                    robustness_demand_routes_copy[i] = new_robustness_demand
                    # 计算新的总体鲁棒性指标
                    new_robustness_indicator = cal_robustness_index(robustness_time_routes_copy, robustness_demand_routes_copy)

                    # 更新最大鲁棒性指标
                    if new_robustness_indicator > max_robustness_indicator or random.random() < 0.05:  # 添加一定的随机性，避免陷入局部最优
                        max_robustness_indicator = new_robustness_indicator
                        best_robustness_time_routes = robustness_time_routes_copy
                        best_robustness_demand_routes = robustness_demand_routes_copy
                        best_route = new_route
                        best_route_index = i
                        inserted = True
                        accepted_move = True
                        break
            if accepted_move:
                break

        if inserted:
            # 更新解
            routes[best_route_index] = best_route
            # 更新robustness_overall_routes
            robustness_time_routes = best_robustness_time_routes
            robustness_demand_routes = best_robustness_demand_routes
        else:
            # 如果没有找到可行位置，则将客户添加到未插入列表
            customer_insert_fail.append(customer)
            continue

    # 如果有客户未插入，则创建新路径
    if customer_insert_fail:
        new_route = [0] + customer_insert_fail + [0]
        routes.append(new_route)

    return routes  # 返回更新后的解

def robustness_focused_local_search_2opt(
    solution: List[List[int]],
    data: Dict[str, Any],
    distance_matrix: np.ndarray,
    travel_time_matrix: np.ndarray,
    travel_time_disturbance: float,
    demand_disturbance: float
) -> List[List[int]]:
    """
    针对鲁棒性目标的2-opt局部搜索操作，遍历所有可能的2-opt交换
    :param solution: 当前解
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
    :param travel_time_matrix: 旅行时间矩阵
    :param travel_time_disturbance: 旅行时间扰动
    :param demand_disturbance: 需求扰动
    :return: 更新后的解
    """
    # 深拷贝解
    routes = copy.deepcopy(solution)

    # 1. 计算每条路径的鲁棒性指标
    time_idx, demand_idx = cal_robustness_index_routes(
        routes, data, travel_time_matrix,
        travel_time_disturbance, demand_disturbance
    )

    # 基准鲁棒性
    baseline = cal_robustness_index(time_idx, demand_idx)

    # first-improvement：记录当前基准方案
    best_gain = baseline
    best_solution = routes
    improved = False

    # 2. 遍历所有路径及所有2-opt交换
    for r_idx, route in enumerate(routes):
        if len(route) <= 3:
            continue
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                # 产生新路径
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                if not check_route_feasible(new_route, data, travel_time_matrix):
                    continue
                # 计算该新路径的鲁棒性
                t_new, d_new = cal_robustness_index_single_route(
                    new_route, data, travel_time_matrix,
                    travel_time_disturbance, demand_disturbance
                )

                # 构造新的总体鲁棒性列表并评估
                temp_time_idx = time_idx.copy()
                temp_demand_idx = demand_idx.copy()
                temp_time_idx[r_idx] = t_new
                temp_demand_idx[r_idx] = d_new
                gain = cal_robustness_index(temp_time_idx, temp_demand_idx)

                # 若有改进，记录最优方案
                if gain > best_gain or random.random() < 0.05:
                    best_gain = gain
                    # 深拷贝当前解并替换改动路径
                    best_solution = copy.deepcopy(routes)
                    best_solution[r_idx] = new_route
                    improved = True
                    break
            if improved:
                break
        if improved:
            break

    return best_solution

def robustness_focused_local_search_oropt(
    solution: List[List[int]],
    data: Dict[str, Any],
    distance_matrix: np.ndarray,
    travel_time_matrix: np.ndarray,
    travel_time_disturbance: float,
    demand_disturbance: float
) -> List[List[int]]:
    """
    针对鲁棒性目标的 Or-OPT 片段重定位局部搜索操作，遍历所有 k=1,2,3 的片段移动
    :param solution: 当前解，每条路径以 0 开头和结束
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
    :param travel_time_matrix: 旅行时间矩阵
    :param travel_time_disturbance: 旅行时间扰动
    :param demand_disturbance: 需求扰动
    :return: 更新后的解
    """
    # 深拷贝解
    routes = copy.deepcopy(solution)

    # 1. 计算全局所有路径的鲁棒性指标
    time_idx, demand_idx = cal_robustness_index_routes(
        routes, data, travel_time_matrix,
        travel_time_disturbance, demand_disturbance
    )
    # 基准鲁棒性
    baseline = cal_robustness_index(time_idx, demand_idx)

    # first-improvement：记录当前基准方案
    best_gain = baseline
    best_solution = routes
    improved = False

    # 2. 遍历所有路径，所有 k, 所有摘取位置 i, 所有插入位置 j
    for r_idx, route in enumerate(routes):
        n = len(route)
        if n <= 4:
            continue
        for k in (1, 2, 3):
            if n <= k + 2:
                break
            # 摘取 segment
            for i in range(1, n - k - 1):
                segment = route[i:i + k]
                remainder = route[:i] + route[i + k:]
                # 插入到 remainder 的每个位置
                for j in range(1, len(remainder)):
                    new_route = remainder[:j] + segment + remainder[j:]
                    if not check_route_feasible(new_route, data, travel_time_matrix):
                        continue
                    # 计算新路径的鲁棒性
                    t_new, d_new = cal_robustness_index_single_route(
                        new_route, data, travel_time_matrix,
                        travel_time_disturbance, demand_disturbance
                    )

                    # 构造新的时间和需求指标列表并评估
                    temp_time_idx = time_idx.copy()
                    temp_demand_idx = demand_idx.copy()
                    temp_time_idx[r_idx] = t_new
                    temp_demand_idx[r_idx] = d_new
                    gain = cal_robustness_index(temp_time_idx, temp_demand_idx)

                    # 若有改进，记录最优方案
                    if gain > best_gain  or random.random() < 0.05:
                        best_gain = gain
                        best_solution = copy.deepcopy(routes)
                        best_solution[r_idx] = new_route
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            break

    return best_solution


def robustness_focused_local_search_exchange(
    solution: List[List[int]],
    data: Dict[str, Any],
    distance_matrix: np.ndarray,
    travel_time_matrix: np.ndarray,
    travel_time_disturbance: float,
    demand_disturbance: float
) -> List[List[int]]:
    """
    针对鲁棒性目标的 Exchange（路径间客户交换）局部搜索算子，
    遍历所有路径对及所有交换位置，记录全局最优鲁棒性改进

    :param solution: 当前解，每条路径以 0 开头和结束
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
    :param travel_time_matrix: 旅行时间矩阵
    :param travel_time_disturbance: 旅行时间扰动
    :param demand_disturbance: 需求扰动
    :return: 更新后的解
    """
    # 深拷贝解
    routes = copy.deepcopy(solution)

    # 1. 计算所有路径鲁棒性指标
    time_idx, demand_idx = cal_robustness_index_routes(
        routes, data, travel_time_matrix,
        travel_time_disturbance, demand_disturbance
    )
    # 基准鲁棒性
    baseline = cal_robustness_index(time_idx, demand_idx)

    # first-improvement：记录当前基准方案
    best_gain = baseline
    best_solution = routes
    improved = False

    K = len(routes)
    # 2. 枚举所有路径对及交换位置
    for p in range(K):
        for q in range(p + 1, K):
            rp, rq = routes[p], routes[q]
            if len(rp) <= 2 or len(rq) <= 2:
                continue

            # 遍历所有客户位置
            for i in range(1, len(rp) - 1):
                for j in range(1, len(rq) - 1):
                    # 生成新路径
                    new_rp, new_rq = rp.copy(), rq.copy()
                    new_rp[i], new_rq[j] = new_rq[j], new_rp[i]

                    if not check_route_feasible(new_rp, data, travel_time_matrix):
                        continue
                    if not check_route_feasible(new_rq, data, travel_time_matrix):
                        continue

                    # 计算新路径的鲁棒性指标
                    t_p, d_p = cal_robustness_index_single_route(
                        new_rp, data, travel_time_matrix,
                        travel_time_disturbance, demand_disturbance
                    )
                    t_q, d_q = cal_robustness_index_single_route(
                        new_rq, data, travel_time_matrix,
                        travel_time_disturbance, demand_disturbance
                    )

                    # 构造更新后的时间和需求指标列表并评估
                    temp_time_idx = time_idx.copy()
                    temp_demand_idx = demand_idx.copy()
                    temp_time_idx[p] = t_p
                    temp_demand_idx[p] = d_p
                    temp_time_idx[q] = t_q
                    temp_demand_idx[q] = d_q
                    gain = cal_robustness_index(temp_time_idx, temp_demand_idx)

                    # 若有改进，记录最优方案
                    if gain > best_gain  or random.random() < 0.05:
                        best_gain = gain
                        best_solution = copy.deepcopy(routes)
                        best_solution[p] = new_rp
                        best_solution[q] = new_rq
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            break

    return best_solution


def robustness_focused_local_search_2opt_star(
    solution: List[List[int]],
    data: Dict[str, Any],
    distance_matrix: np.ndarray,
    travel_time_matrix: np.ndarray,
    travel_time_disturbance: float,
    demand_disturbance: float
) -> List[List[int]]:
    """
    针对鲁棒性目标的 2-opt*（交叉交换尾段）局部搜索算子，
    遍历所有路径对及所有尾段切点，记录并应用全局最优鲁棒性改进。

    :param solution: 当前解，每条路径以 0 开头和结束
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
    :param travel_time_matrix: 旅行时间矩阵
    :param travel_time_disturbance: 旅行时间扰动
    :param demand_disturbance: 需求扰动
    :return: 更新后的解
    """
    # 深拷贝解
    routes = copy.deepcopy(solution)

    # 1. 计算所有路径的鲁棒性指标
    time_idx, demand_idx = cal_robustness_index_routes(
        routes, data, travel_time_matrix,
        travel_time_disturbance, demand_disturbance
    )
    # 基准鲁棒性
    baseline = cal_robustness_index(time_idx, demand_idx)

    # first-improvement：记录当前基准方案
    best_gain = baseline
    best_solution = routes
    improved = False

    K = len(routes)
    # 2. 枚举所有路径对及尾段切点 (i,j)
    for p in range(K):
        for q in range(p + 1, K):
            rp, rq = routes[p], routes[q]
            # 确保每条路径至少 depot+1 客户+depot
            if len(rp) <= 3 or len(rq) <= 3:
                continue


            # i 从 1 到 len(rp)-2, j 从 1 到 len(rq)-2
            for i in range(1, len(rp) - 1):
                for j in range(1, len(rq) - 1):
                    # 交叉交换尾段
                    new_rp = rp[:i] + rq[j:]
                    new_rq = rq[:j] + rp[i:]

                    if not check_route_feasible(new_rp, data, travel_time_matrix):
                        continue
                    if not check_route_feasible(new_rq, data, travel_time_matrix):
                        continue

                    # 计算新路径的鲁棒性指标
                    t_p, d_p = cal_robustness_index_single_route(
                        new_rp, data, travel_time_matrix,
                        travel_time_disturbance, demand_disturbance
                    )
                    t_q, d_q = cal_robustness_index_single_route(
                        new_rq, data, travel_time_matrix,
                        travel_time_disturbance, demand_disturbance
                    )

                    # 构造更新后的时间和需求指标列表并评估
                    temp_time_idx = time_idx.copy()
                    temp_demand_idx = demand_idx.copy()
                    temp_time_idx[p] = t_p
                    temp_demand_idx[p] = d_p
                    temp_time_idx[q] = t_q
                    temp_demand_idx[q] = d_q
                    gain = cal_robustness_index(temp_time_idx, temp_demand_idx)

                    # 若有改进，记录最优方案
                    if gain > best_gain  or random.random() < 0.05:
                        best_gain = gain
                        best_solution = copy.deepcopy(routes)
                        best_solution[p] = new_rp
                        best_solution[q] = new_rq
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            break


    return best_solution

def robustness_focused_local_search_cross_exchange(
    solution: List[List[int]],
    data: Dict[str, Any],
    distance_matrix: np.ndarray,
    travel_time_matrix: np.ndarray,
    travel_time_disturbance: float,
    demand_disturbance: float
) -> List[List[int]]:
    """
    针对鲁棒性目标的 Cross-Exchange（跨路径片段交换）局部搜索算子，
    遍历所有路径对、所有片段长度和位置组合，记录并应用全局最优鲁棒性改进。

    :param solution: 当前解，每条路径以 0 开头和结束
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
    :param travel_time_matrix: 旅行时间矩阵
    :param travel_time_disturbance: 旅行时间扰动
    :param demand_disturbance: 需求扰动
    :return: 更新后的解
    """
    # 深拷贝解
    routes = copy.deepcopy(solution)

    # 1. 计算所有路径的鲁棒性指标
    time_idx, demand_idx = cal_robustness_index_routes(
        routes, data, travel_time_matrix,
        travel_time_disturbance, demand_disturbance
    )
    # 基准鲁棒性
    baseline = cal_robustness_index(time_idx, demand_idx)

    # first-improvement：记录当前基准方案
    best_gain = baseline
    best_solution = routes
    improved = False

    K = len(routes)
    # 2. 遍历所有路径对及片段长度 k
    for p in range(K):
        for q in range(p + 1, K):
            rp, rq = routes[p], routes[q]
            # 每条路径至少保留 depot+1 客户+depot
            if len(rp) <= 3 or len(rq) <= 3:
                continue

            for k in (1, 2, 3):
                # 确保可摘取
                if len(rp) <= k + 1 or len(rq) <= k + 1:
                    break

                # 在 rp 上摘段 [i:i+k]
                for i in range(1, len(rp) - k):
                    # 在 rq 上摘段 [j:j+k]
                    for j in range(1, len(rq) - k):
                        seg_p = rp[i:i + k]
                        rem_p = rp[:i] + rp[i + k:]
                        seg_q = rq[j:j + k]
                        rem_q = rq[:j] + rq[j + k:]

                        # 生成交换后路径
                        new_p = rem_p[:i] + seg_q + rem_p[i:]
                        new_q = rem_q[:j] + seg_p + rem_q[j:]

                        if not check_route_feasible(new_p, data, travel_time_matrix):
                            continue
                        if not check_route_feasible(new_q, data, travel_time_matrix):
                            continue

                        # 计算新路径的鲁棒性指标
                        t_p, d_p = cal_robustness_index_single_route(
                            new_p, data, travel_time_matrix,
                            travel_time_disturbance, demand_disturbance
                        )
                        t_q, d_q = cal_robustness_index_single_route(
                            new_q, data, travel_time_matrix,
                            travel_time_disturbance, demand_disturbance
                        )

                        # 构造更新后的时间和需求指标列表并评估
                        temp_time_idx = time_idx.copy()
                        temp_demand_idx = demand_idx.copy()
                        temp_time_idx[p] = t_p
                        temp_demand_idx[p] = d_p
                        temp_time_idx[q] = t_q
                        temp_demand_idx[q] = d_q
                        gain = cal_robustness_index(temp_time_idx, temp_demand_idx)

                        # 若有改进，记录最优方案
                        if gain > best_gain:
                            best_gain = gain
                            best_solution = copy.deepcopy(routes)
                            best_solution[p] = new_p
                            best_solution[q] = new_q
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

    # # 3. 应用最优改进（若存在）
    # if best_gain > baseline or random.random() < 0.05:
    return best_solution



def check_route_feasible(route: List[int], data: Dict[str, Any], travel_time_matrix: np.ndarray) -> bool:
    '''
    检查路径是否可行
    :param route: 路径
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
    :param max_vehicle: 最大车辆数量
    :param capacity: 车辆容量
    :return: 是否可行
    '''

    load = 0
    capacity = data['capacity']
    current_time = 0
    for i in range(1, len(route) - 1):
        customer = route[i]
        load += data['DEMAND'][customer]
        if load > capacity:
            return False

        travel_time = travel_time_matrix[route[i - 1], customer]
        arrival_time = current_time + travel_time
        service_start_time = max(arrival_time, data['READY_TIME'][customer])
        if service_start_time > data['DUE_DATE'][customer]:
            return False
        current_time = service_start_time + data['SERVICE_TIME'][customer]
        
    return True