import copy
import random
import math


def _sample_bounded_disturbance(max_abs_disturbance):
    limit = abs(max_abs_disturbance)
    return random.uniform(-limit, limit)


def _keep_positive(value, minimum):
    return max(value, minimum)


def add_stochastic_disturbance(problem_data, distance_matrix, travel_time_disturbance=3, demand_disturbance=3):
    """
    添加随机扰动
    :param problem_data: 问题数据
    :param distance_matrix: 距离矩阵
    :param stochastic_disturbance: 随机干扰
    :param demand_disturbance: 需求扰动
    :return: 更新后的问题数据和距离矩阵
    """
    real_travel_time = copy.deepcopy(distance_matrix)
    for i in range(len(real_travel_time)):
        for j in range(i + 1, len(real_travel_time[i])):
            disturbance = _sample_bounded_disturbance(travel_time_disturbance)
            real_travel_time[i][j] = _keep_positive(real_travel_time[i][j] + disturbance, 0.1)
            real_travel_time[j][i] = _keep_positive(real_travel_time[j][i] + disturbance, 0.1)  # 保持对称性

    real_problem_data = copy.deepcopy(problem_data)
    for customer in range(1, len(real_problem_data['DEMAND'])):
        # 添加需求扰动
        demand_disturbance_value = _sample_bounded_disturbance(demand_disturbance)
        real_problem_data['DEMAND'][customer] = _keep_positive(
            real_problem_data['DEMAND'][customer] + demand_disturbance_value,
            1,
        )

    return real_problem_data, real_travel_time

def add_stochastic_disturbance_all_customers(all_customers_data, travel_time_disturbance=3, demand_disturbance=3, travel_time_matrix=None):
    """
    为所有客户数据添加随机扰动
    :param all_customers_data: 所有客户数据
    :param travel_time_disturbance: 旅行时间扰动
    :param demand_disturbance: 需求扰动
    :param travel_time_matrix: 旅行时间矩阵
    :return: 添加扰动后的客户数据
    """
    real_all_customers_data = copy.deepcopy(all_customers_data)
    for customer in range(1, len(real_all_customers_data)):
        # 添加需求扰动
        demand_disturbance_value = _sample_bounded_disturbance(demand_disturbance)
        real_all_customers_data.loc[customer, 'DEMAND'] = _keep_positive(
            real_all_customers_data.loc[customer, 'DEMAND'] + demand_disturbance_value,
            0.1,
        )


    # 计算travel time矩阵
    if travel_time_matrix is None:
        travel_time = [[0] * len(real_all_customers_data) for _ in range(len(real_all_customers_data))]
        for i in range(len(real_all_customers_data)):
            for j in range(len(real_all_customers_data)):
                if i != j:
                    xi, yi = real_all_customers_data['XCOORD'][i], real_all_customers_data['YCOORD'][i]
                    xj, yj = real_all_customers_data['XCOORD'][j], real_all_customers_data['YCOORD'][j]
                    distance = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    disturbance = _sample_bounded_disturbance(travel_time_disturbance)
                    travel_time[i][j] = _keep_positive(distance + disturbance, 0.1)
    else:
        travel_time = copy.deepcopy(travel_time_matrix)
        for i in range(len(travel_time)):
            for j in range(len(travel_time[i])):
                if i != j:
                    disturbance = _sample_bounded_disturbance(travel_time_disturbance)
                    travel_time[i][j] = _keep_positive(travel_time[i][j] + disturbance, 0.1)
    return real_all_customers_data, travel_time

def robustness_check(vehicle_data, real_problem_data, real_travel_time):
    """
    检查鲁棒性，计算多次测试的平均延迟时间和容量违规
    :param vehicle_data: 车辆数据
    :param problem_data: 问题数据
    :param real_distance_matrix: 真实距离矩阵
    :param travel_time_disturbance: 旅行时间扰动
    :param demand_disturbance: 需求扰动
    :return: 实际延迟时间和容量违规
    """
    capacity = real_problem_data['capacity']
    total_delay_time = 0
    total_capacity_violation = 0
    for vehicle in vehicle_data:
        route = vehicle['served_customers']
        current_time = 0
        capacity_used = 0
        for i in range(1, len(route)):
            customer = route[i]
            previous_customer = route[i - 1]
            ready_time = real_problem_data['READY_TIME'][customer]
            due_date = real_problem_data['DUE_DATE'][customer]
            service_time = real_problem_data['SERVICE_TIME'][customer]
            # 计算旅行时间
            travel_time = real_travel_time[previous_customer][customer]
            arrive_time = current_time + travel_time
            # 计算延迟时间
            delay_time = max(0, arrive_time - due_date)
            total_delay_time += delay_time
            # 更新当前时间
            current_time = max(arrive_time, ready_time) + service_time
            if customer != 0:
                capacity_used += real_problem_data['DEMAND'][customer]

        total_capacity_violation += max(0, capacity_used - capacity)
    return total_delay_time, total_capacity_violation

def robustness_check_routes(routes, disturbed_problem_data, disturbed_travel_time):
    """
    检查给定路线的鲁棒性，计算平均延迟时间和容量违规
    :param routes: 路线列表
    :param disturbed_problem_data: 扰动后的问题数据
    :param disturbed_travel_time: 扰动后的旅行时间矩阵
    :return: 实际延迟时间和容量违规
    """
    total_delay_time = 0
    total_capacity_violation = 0
    capacity = disturbed_problem_data['capacity']

    for route in routes:
        current_time = 0
        capacity_used = 0
        
        for i in range(1, len(route)):
            customer = route[i]
            previous_customer = route[i - 1]
            ready_time = disturbed_problem_data['READY_TIME'][customer]
            due_date = disturbed_problem_data['DUE_DATE'][customer]
            service_time = disturbed_problem_data['SERVICE_TIME'][customer]
            travel_time = disturbed_travel_time[previous_customer][customer]

            arrive_time = current_time + travel_time
            delay_time = max(0, arrive_time - due_date)
            total_delay_time += delay_time
            
            current_time = max(arrive_time, ready_time) + service_time
            if customer != 0:
                capacity_used += disturbed_problem_data['DEMAND'][customer]

        total_capacity_violation += max(0, capacity_used - capacity)

    return total_delay_time, total_capacity_violation

def robustness_test(vehicle_data, problem_data, real_distance_matrix, travel_time_disturbance=3, demand_disturbance=3, test_times=50):
    """
    进行鲁棒性测试，计算多次测试的平均延迟时间
    :param vehicle_data: 车辆数据
    :param problem_data: 问题数据
    :param real_distance_matrix: 真实距离矩阵
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
        for vehicle in vehicle_data:
            route = vehicle['served_customers']
            total_delay = 0
            total_capacity_used = 0
            current_time = 0
            for i in range(1, len(route)):
                customer = route[i]
                previous_customer = route[i - 1]
                travel_time = _keep_positive(
                    real_distance_matrix[previous_customer][customer] + _sample_bounded_disturbance(travel_time_disturbance),
                    1,
                )

                arrive_time = current_time + travel_time
                ready_time = problem_data['READY_TIME'][customer]
                due_date = problem_data['DUE_DATE'][customer]
                service_time = problem_data['SERVICE_TIME'][customer]

                total_delay += max(0, arrive_time - due_date)
                current_time = max(arrive_time, ready_time) + service_time

                if customer != 0:
                    demand = _keep_positive(
                        problem_data['DEMAND'][customer] + _sample_bounded_disturbance(demand_disturbance),
                        0.1,
                    )
                    total_capacity_used += demand

            total_delay_time += total_delay
            total_capacity_violation += max(0, total_capacity_used - capacity)
        all_tests_delay_time += total_delay_time
        all_tests_capacity_violation += total_capacity_violation
    # 计算平均延迟时间
    total_delay_time = all_tests_delay_time / test_times
    # 计算平均容量违规
    total_capacity_violation = all_tests_capacity_violation / test_times

    return total_delay_time, total_capacity_violation
