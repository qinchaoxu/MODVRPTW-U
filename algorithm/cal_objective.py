'''
Description: 计算目标函数值，规定所有路径方案为一个二维数组，每个路径方案为一个一维数组，每个一维数组中的元素为客户编号，所有路径以0开始，0结束
''' 

import numpy as np
import pandas as pd
from algorithm.benchmark_process import calculate_distance, calculate_time, load_data, cal_distance_matrix
from algorithm.cal_std_index import cal_stdi

def cal_total_cost(distance_matrix, route_plan):
    '''
    计算路径方案的总成本，即总距离
    :param distance_matrix: 距离矩阵
    :param route_plan: 路径方案
    :return: 路径方案的总成本
    '''
    total_cost = 0
    for route in route_plan:
        total_cost += calculate_distance(distance_matrix, route)
    return total_cost

def cal_total_travel_time(data, distance_matrix, route_plan, speed=1):
    '''
    计算路径方案的总旅行时间
    :param distance_matrix: 距离矩阵
    :param route_plan: 路径方案
    :param speed: 车辆速度
    :return: 路径方案的总旅行时间
    '''
    total_time = 0
    for route in route_plan:
        total_time += calculate_time(data, distance_matrix, route, speed)[-1]
    return total_time

def cal_total_delay_time(data, distance_matrix, route_plan):
    '''
    计算路径方案的总延误时间
    :param distance_matrix: 距离矩阵
    :param route_plan: 路径方案
    :param data: 客户数据
    :return: 路径方案的总延误时间
    '''
    total_delay_time = 0
    for route in route_plan:
        time_list = calculate_time(data, distance_matrix, route)[1:]
        delay = time_list - data['DUE_DATE'][route[1:]]
        total_delay_time += np.sum(delay[delay > 0])
    return total_delay_time

def cal_total_travel_time_and_delay_time(data, distance_matrix, route_plan):
    '''
    同时计算路径方案的总旅行时间和总延误时间
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
    :param route_plan: 路径方案
    '''
    total_travel_time = 0
    total_delay_time = 0
    for route in route_plan:
        time_list = calculate_time(data, distance_matrix, route)[1:]
        total_travel_time += calculate_time(data, distance_matrix, route)[-1]
        delay = time_list - data['DUE_DATE'][route[1:]]
        total_delay_time += np.sum(delay[delay > 0])
    return total_travel_time, total_delay_time

def cal_total_distance_and_travel_time_and_delay_time(data, distance_matrix, route_plan, speed=1):
    '''
    同时计算路径方案的总距离、总旅行时间和总延误时间
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
    :param route_plan: 路径方案
    :param speed: 车辆速度
    '''
    # 预提取数组
    ready   = data['READY_TIME']
    service = data['SERVICE_TIME']
    due     = data['DUE_DATE']

    total_cost = total_time = total_delay = 0.0

    for route in route_plan:
        t = 0.0
        for prev, cur in zip(route[:-1], route[1:]):
            d = distance_matrix[prev, cur]
            total_cost += d

            travel = d / speed
            t_arr = t + travel
            total_delay += max(0.0, t_arr - due[cur])

            # 更新服务完毕后的时刻
            t = max(t_arr, ready[cur]) + service[cur]

        total_time += t_arr

    return total_cost, total_time, total_delay

def cal_total_distance_and_wait_time_and_delay_time(data, distance_matrix, travel_time_matrix, route_plan):
    '''
    同时计算路径方案的总距离、总等待时间和总延误时间
    :param data: 客户数据
    :param distance_matrix: 距离矩阵
    :param travel_time_matrix: 旅行时间矩阵
    :param route_plan: 路径方案
    '''
    # 预提取数组
    ready   = data['READY_TIME']
    service = data['SERVICE_TIME']
    due     = data['DUE_DATE']

    total_cost = total_delay = total_wait = 0.0

    for route in route_plan:
        t = 0.0
        for prev, cur in zip(route[:-1], route[1:]):
            d = distance_matrix[prev, cur]
            total_cost += d

            travel_time = travel_time_matrix[prev, cur]
            t_arr = t + travel_time
            total_wait += max(0.0, ready[cur] - t_arr)
            total_delay += max(0.0, t_arr - due[cur])

            # 更新服务完毕后的时刻
            t = max(t_arr, ready[cur]) + service[cur]

    return total_cost, total_wait, total_delay

def cal_vehicle_num(route_plan):
    '''
    计算路径方案的车辆数
    :param route_plan: 路径方案
    :return: 路径方案的车辆数
    '''
    return len(route_plan)

def cal_stdi_obj(data, travel_time_matrix, route_plan):
    '''
    计算路径方案 STDI 目标函数值，即对 STDI 取相反数用于最小化
    :param data: 客户数据
    :param travel_time_matrix: 旅行时间矩阵
    :param route_plan: 路径方案
    :return: 路径方案的 STDI 目标函数值
    '''
    stdi = cal_stdi(route_plan, data, travel_time_matrix)
    return -stdi

