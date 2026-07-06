'''
Description: 计算路径方案的 Slack Time Distribution Index (STDI).
'''

import numpy as np
from algorithm.benchmark_process import calculate_arrive_time


def find_slack_times(route_plan, data, travel_time_matrix):
    '''
    计算路径方案中每个客户的 slack time.
    
    论文中 sigma_i = max(e_i - t_i, 0), 其中 e_i 是客户最早服务时间,
    t_i 是车辆到达客户 i 的时间。这里只统计客户节点, 不统计 depot.
    '''
    slack_times_by_route = []
    
    for route in route_plan:
        route_slack_times = []
        time_list = calculate_arrive_time(data, travel_time_matrix, route)

        for i in range(1, len(route)):
            customer = route[i]
            if customer == 0:
                continue

            ready_time = data['READY_TIME'][customer]
            arrival_time = time_list[i]
            route_slack_times.append(max(ready_time - arrival_time, 0.0))
        
        slack_times_by_route.append(route_slack_times)

    return slack_times_by_route


def cal_stdi(route_plan, data, travel_time_matrix, mode='previous', theta=0.5):
    '''
    计算论文中的 Slack Time Distribution Index (STDI).

    :param route_plan: 路径方案
    :param data: 客户数据
    :param travel_time_matrix: 旅行时间矩阵
    :return: STDI
    '''
    slack_times_by_route = find_slack_times(route_plan, data, travel_time_matrix)
    return cal_stdi_from_slacks(slack_times_by_route, data)

def cal_stdi_from_slacks(slack_times_by_route, data, mode='previous', theta=0.5):
    '''
    根据已缓存的 route-level slack times 计算 STDI.
    '''
    used_vehicle_num = sum(1 for route_slacks in slack_times_by_route if len(route_slacks) > 0)
    if used_vehicle_num == 0:
        return 0

    slack_times = [item for route_slacks in slack_times_by_route for item in route_slacks]
    if len(slack_times) == 0:
        return 0

    slack_times = np.array(slack_times, dtype=float)
    total_slack = np.sum(slack_times)
    if total_slack <= 0:
        return 0

    mean_slack = np.mean(slack_times)
    sigma_cv = np.sqrt(np.mean((slack_times - mean_slack) ** 2)) / mean_slack

    t_max = data['DUE_DATE'][0]
    if t_max <= 0:
        return 0

    normalized_total_slack = total_slack / (used_vehicle_num * t_max)
    uniformity = 1 / (1 + sigma_cv)

    return normalized_total_slack * uniformity
