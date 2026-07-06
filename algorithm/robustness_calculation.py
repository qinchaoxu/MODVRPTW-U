import numpy as np
import random


def _sample_bounded_disturbance(max_abs_disturbance):
    limit = abs(max_abs_disturbance)
    return random.uniform(-limit, limit)


def cal_robustness_index_routes(route_plan, data, travel_time_matrix, travel_time_disturbance=0, demand_disturbance=0):
    '''
    计算路径方案的鲁棒性指标
    :param route_plan: 路径方案
    :param data: 客户数据
    :param travel_time_matrix: 旅行时间矩阵
    :param travel_time_disturbance: 旅行时间扰动
    :param demand_disturbance: 需求扰动
    :return: 每条路径包含的每个客户的时间鲁棒性指标和需求鲁棒性指标
    '''

    robustness_time_routes = []
    robustness_demand_routes = []
    for route in route_plan:
        # 计算每条路径的鲁棒性指标
        current_time = 0
        demand = 0
        robustness_time_customers = []

        for i in range(1, len(route)):

            customer = route[i]
            previous_customer = route[i - 1]

            travel_time = travel_time_matrix[previous_customer][customer]
            arrive_time = current_time + travel_time + travel_time_disturbance
            ready_time = data['READY_TIME'][customer]
            due_date = data['DUE_DATE'][customer]
            service_time = data['SERVICE_TIME'][customer]

            if customer != 0:
                if travel_time_disturbance == 0:
                    robustness_time = 1.0
                else:
                    robustness_time = max(min((due_date - arrive_time) / travel_time_disturbance, 1), 0)
                robustness_time_customers.append(robustness_time)

            current_time = max(arrive_time, ready_time) + service_time
            if customer != 0:
                demand += data['DEMAND'][customer]

        if demand_disturbance == 0:
            robustness_demand = 1.0
        else:
            customer_count = sum(1 for customer in route if customer != 0)
            if customer_count == 0:
                robustness_demand = 1.0
            else:
                robustness_demand = max(min((data['capacity'] - demand) / (customer_count * demand_disturbance), 1), 0)
        # 时间鲁棒性指标直接保存每个客户的指标
        robustness_time_routes.append(robustness_time_customers)
        robustness_demand_routes.append(robustness_demand)

    return robustness_time_routes, robustness_demand_routes

def cal_robustness_index_single_route(route, data, travel_time_matrix, travel_time_disturbance=0, demand_disturbance=0):
    '''
    计算单条路径的鲁棒性指标
    :param route: 路径
    :param data: 客户数据
    :param travel_time_matrix: 旅行时间矩阵
    :param travel_time_disturbance: 旅行时间扰动
    :param demand_disturbance: 需求扰动
    :return: 每条路径的时间鲁棒性指标和需求鲁棒性指标
    '''

    # 计算单条路径的鲁棒性指标
    current_time = 0
    demand = 0
    robustness_time_customers = []

    for i in range(1, len(route)):

        customer = route[i]
        previous_customer = route[i - 1]

        travel_time = travel_time_matrix[previous_customer][customer]
        arrive_time = current_time + travel_time + travel_time_disturbance
        ready_time = data['READY_TIME'][customer]
        due_date = data['DUE_DATE'][customer]
        service_time = data['SERVICE_TIME'][customer]

        if customer != 0:
            if travel_time_disturbance == 0:
                robustness_time = 1.0
            else:
                robustness_time = max(min((due_date - arrive_time) / travel_time_disturbance, 1), 0)
            robustness_time_customers.append(robustness_time)

        current_time = max(arrive_time, ready_time) + service_time
        if customer != 0:
            demand += data['DEMAND'][customer]
    if demand_disturbance == 0:
        robustness_demand_route = 1.0
    else:
        customer_count = sum(1 for customer in route if customer != 0)
        if customer_count == 0:
            robustness_demand_route = 1.0
        else:
            robustness_demand_route = max(min((data['capacity'] - demand) / (customer_count * demand_disturbance), 1), 0)
    # 时间鲁棒性指标直接保存每个客户的指标
    robustness_time_route = robustness_time_customers

    return robustness_time_route, robustness_demand_route

def cal_robustness_index(robustness_time_routes, robustness_demand_routes):
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
    return (final_robustness_time + final_robustness_demand) / 2

def cal_robustness_index_devide(robustness_time_routes, robustness_demand_routes):
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
    return final_robustness_time, final_robustness_demand

def robustness_test_routes(route_plan, problem_data, travel_time_matrix, travel_time_disturbance=3, demand_disturbance=3, test_times=50):
    """
    进行鲁棒性测试，计算多次测试的平均延迟时间
    :param vehicle_data: 车辆数据
    :param problem_data: 问题数据
    :param travel_time_matrix: 真实旅行时间矩阵
    :param travel_time_disturbance: 旅行时间扰动
    :param demand_disturbance: 需求扰动
    :return: 实际延迟时间
    """
    capacity = problem_data['capacity']
    all_tests_delay_time = 0
    all_tests_capacity_violation = 0

    # 进行多次测试以计算平均延迟时间与容量违规
    for _ in range(test_times):

        total_delay_time = 0
        total_capacity_violation = 0
        for route in route_plan:
            total_delay = 0
            total_capacity_used = 0
            current_time = 0
            for i in range(1, len(route)):
                customer = route[i]
                previous_customer = route[i - 1]
                travel_time = travel_time_matrix[previous_customer][customer] + _sample_bounded_disturbance(travel_time_disturbance)
                if travel_time < 0:
                    travel_time = 1  # 确保旅行时间不为负数

                arrive_time = current_time + travel_time
                ready_time = problem_data['READY_TIME'][customer]
                due_date = problem_data['DUE_DATE'][customer]
                service_time = problem_data['SERVICE_TIME'][customer]

                total_delay += max(0, arrive_time - due_date)
                current_time = max(arrive_time, ready_time) + service_time

                if customer != 0:
                    demand = problem_data['DEMAND'][customer] + _sample_bounded_disturbance(demand_disturbance)
                    total_capacity_used += max(demand, 0.1)

            total_delay_time += total_delay
            total_capacity_violation += max(0, total_capacity_used - capacity)
        all_tests_delay_time += total_delay_time
        all_tests_capacity_violation += total_capacity_violation
    # 计算平均延迟时间
    total_delay_time = all_tests_delay_time / test_times
    # 计算平均容量违规
    total_capacity_violation = all_tests_capacity_violation / test_times

    return total_delay_time, total_capacity_violation
