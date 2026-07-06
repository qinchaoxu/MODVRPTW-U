'''
Description: 这个文件主要根据SOLOMON数据集生成动态客户数据，通过均匀随机分布选取后续到达的客户和其出现时间
'''

import random
import numpy as np
import copy
from algorithm.neighborhood_search_robust import neighborhood_search_robust


def generate_dynamic_customers_one(data, dynamic_customers_ratio=0.3):
    '''
    生成动态客户数据
    :param data: 原始客户数据，包含坐标、时间窗等信息的DataFrame
    :param dynamic_customers_ratio: 动态客户比例，表示在每个时间段内选择的动态客户数量占总客户数量的比例
    :return: 整个客户数据，增加一列表示动态客户的出现时间，静态客户为0
    '''
    # 计算动态客户数量
    total_customers = len(data) - 1  # 减去仓库客户
    dynamic_customers_count = int(total_customers * dynamic_customers_ratio)
    
    # 随机选择动态客户
    dynamic_customers_indices = random.sample(range(1, total_customers + 1), dynamic_customers_count)

    # # 创建动态客户数据
    all_customers_data = data.copy()
    # dynamic_customers_data = data.iloc[dynamic_customers_indices].copy()
    
    # 生成动态客户的出现时间和服务时间
    for index, row in all_customers_data.iterrows():
        # 增加一列表示到达时间
        if index in dynamic_customers_indices:
            # 如果是动态客户，随机选择到达时间
            if row['READY_TIME'] == 0:
                arrival_time = random.randint(int(row['READY_TIME']*0.1), int(row['DUE_DATE']*0.2))
            else:
                arrival_time = random.randint(int(row['READY_TIME']*0.1), int(row['READY_TIME']*0.75))
        else:
            # 如果是静态客户，到达时间为0
            arrival_time = 0
        all_customers_data.at[index, 'ARRIVAL_TIME'] = arrival_time

    return all_customers_data.reset_index(drop=True)

def update_data(problem_data, distance_matrix, new_customer_data):
    '''
    更新问题数据和距离矩阵
    :param problem_data: 问题数据
    :param distance_matrix: 原始距离矩阵
    :param new_customer_data: 新客户数据
    :return: 更新后的问题数据和距离矩阵
    '''
    # 计算新客户与所有客户之间的距离
    new_distances = np.sqrt((problem_data['XCOORD'] - new_customer_data['XCOORD'])**2 + (problem_data['YCOORD'] - new_customer_data['YCOORD'])**2)

    # 更新距离矩阵
    distance_matrix = np.vstack((distance_matrix, new_distances))
    distance_matrix = np.hstack((distance_matrix, np.append(new_distances, 0).reshape(-1, 1)))

    # 更新问题数据
    problem_data['XCOORD'] = np.append(problem_data['XCOORD'], new_customer_data['XCOORD'])
    problem_data['YCOORD'] = np.append(problem_data['YCOORD'], new_customer_data['YCOORD'])
    problem_data['DEMAND'] = np.append(problem_data['DEMAND'], new_customer_data['DEMAND'])
    problem_data['READY_TIME'] = np.append(problem_data['READY_TIME'], new_customer_data['READY_TIME'])
    problem_data['DUE_DATE'] = np.append(problem_data['DUE_DATE'], new_customer_data['DUE_DATE'])
    problem_data['SERVICE_TIME'] = np.append(problem_data['SERVICE_TIME'], new_customer_data['SERVICE_TIME'])

    return problem_data, distance_matrix

def update_disturbed_data(disturbed_problem_data, new_customer_data, disturbed_customers_data, disturbed_travel_time_all, arrived_customers):
    '''
    更新扰动后的问题数据和距离矩阵
    :param disturbed_problem_data: 扰动后的问题数据
    :param disturbed_travel_time: 扰动后的旅行时间矩阵
    :param new_customer_data: 新客户数据
    :param disturbed_customers_data: 扰动后的所有客户数据
    :param disturbed_travel_time_all: 扰动后的旅行时间矩阵（包含所有客户）
    :return: 更新后的问题数据和旅行时间矩阵
    '''
    new_customer_index = new_customer_data['CUST_NO']
    disturbed_problem_data['XCOORD'] = np.append(disturbed_problem_data['XCOORD'], new_customer_data['XCOORD'])
    disturbed_problem_data['YCOORD'] = np.append(disturbed_problem_data['YCOORD'], new_customer_data['YCOORD'])
    disturbed_problem_data['DEMAND'] = np.append(disturbed_problem_data['DEMAND'], disturbed_customers_data.loc[new_customer_index, 'DEMAND'])
    disturbed_problem_data['READY_TIME'] = np.append(disturbed_problem_data['READY_TIME'], new_customer_data['READY_TIME'])
    disturbed_problem_data['DUE_DATE'] = np.append(disturbed_problem_data['DUE_DATE'], new_customer_data['DUE_DATE'])
    disturbed_problem_data['SERVICE_TIME'] = np.append(disturbed_problem_data['SERVICE_TIME'], new_customer_data['SERVICE_TIME'])

    disturbed_travel_time = np.array([[disturbed_travel_time_all[i][j] for j in arrived_customers] for i in arrived_customers])  # 只保留静态客户之间的距离

    return disturbed_problem_data, disturbed_travel_time


def insert_dynamic_customers_robust(vehicle_data, dynamic_customers_data, current_time, problem_data, distance_matrix, delay_tolerance_time, real_problem_data=None, travel_time_disturbance=0, demand_disturbance=0, v=1, iteration_max=100, p_kr=0.2):
    '''
    将动态客户插入到车辆的路径中
    :param vehicle_data: 车辆数据
    :param dynamic_customers_data: 动态客户数据
    :param current_time: 当前时间
    :param problem_data: 问题数据
    :param distance_matrix: 距离矩阵
    :param delay_tolerance_time: 延误容忍时间
    :param v: 可选参数，速度
    '''
    vehicle_data_uninserted = copy.deepcopy(vehicle_data)  # 深拷贝车辆数据，避免在遍历时修改原数据

    new_customer = len(problem_data['XCOORD']) - 1 # 新客户的ID为当前问题数据中客户数量(不包括仓库)的最后一个索引
    new_customer_location = (dynamic_customers_data['XCOORD'], dynamic_customers_data['YCOORD'])
    new_customer_ready_time = dynamic_customers_data['READY_TIME']
    new_customer_due_date = dynamic_customers_data['DUE_DATE']

    # 遍历每个车辆的路径，找到距离新客户时空距离最近的位置
    min_distance_feasible = float('inf')  # 最小可行距离
    best_vehicle_id_feasible = None
    min_distance_infeasible = float('inf') # 最小不可行距离
    for vehicle in vehicle_data:
        route = vehicle['unserved_customers'].copy()  # 获取车辆的未服务客户路径
        if len(route) <= 1:
            # 如果车辆没有未服务的客户或者只剩一个客户（仓库），则跳过
            continue
        # 计算车辆目前已用的容量
        if real_problem_data != None:
            current_capacity_used = sum(real_problem_data['DEMAND'][customer] for customer in vehicle['served_customers'])
        else:
            current_capacity_used = sum(problem_data['DEMAND'][customer] for customer in vehicle['served_customers'])
        # 遍历当前车辆的未服务客户，找到新客户可以插入的位置
        last_customer = vehicle['current_customer']  # 该车辆当前服务的客户，即unserved_customers的第一个客户
        last_service_end_time = vehicle['expected_service_end_time']  # 当前服务的客户预计服务结束时间
        leave_last_customer_time = last_service_end_time  # 离开当前服务客户的时间，所有插入都以该时间为基准，因为目前策略车辆服务结束后直接前往下一个客户，所以直接等于 last_service_end_time
        route_start_time = leave_last_customer_time
        for i in range(1, len(route)):
            current_customer = route[i]  # 当前客户
            inserted_route = route[:i] + [new_customer] + route[i:]  # 在第i个位置插入新客户，即当前客户之前
            # 计算插入点前一客户到达新插入客户的时间
            arrive_new_customer_time = leave_last_customer_time + distance_matrix[last_customer, new_customer]
            # 计算当前客户点(x,y,arrive_new_customer_time)到新客户(x_new,y_new,ready_time)的时空距离
            distance_to_new_customer = np.sqrt((problem_data['XCOORD'][last_customer] - new_customer_location[0])**2
                                                + (problem_data['YCOORD'][last_customer] - new_customer_location[1])**2
                                                  + (v*(arrive_new_customer_time - new_customer_ready_time))**2)
            # 检查新插入的路径是否可行
            feasibility = check_feasibility_robust(inserted_route, route_start_time, problem_data, distance_matrix, current_capacity_used, travel_time_disturbance, demand_disturbance)
            if feasibility:
                # 如果可行，检查是否是最小可行距离
                if distance_to_new_customer < min_distance_feasible:
                    min_distance_feasible = distance_to_new_customer
                    best_vehicle_id_feasible = vehicle['vehicle_id']
                    best_insert_index_feasible = i
                    best_inserted_route_feasible = inserted_route
            else:
                # 如果不可行，检查是否是最小不可行距离
                if distance_to_new_customer < min_distance_infeasible:
                    min_distance_infeasible = distance_to_new_customer
                    best_vehicle_id_infeasible = vehicle['vehicle_id']
                    best_insert_index_infeasible = i
                    best_inserted_route_infeasible = inserted_route
            arrive_current_customer_time = leave_last_customer_time + distance_matrix[last_customer, current_customer]
            leave_current_customer_time = max(arrive_current_customer_time, problem_data['READY_TIME'][current_customer]) + problem_data['SERVICE_TIME'][current_customer]
            # 更新last_customer和leave_last_customer_time
            last_customer = route[i]
            leave_last_customer_time = leave_current_customer_time

    if best_vehicle_id_feasible is not None:
        # 如果找到了可行的插入位置，则将新客户插入到该位置
        vehicle = next(v for v in vehicle_data if v['vehicle_id'] == best_vehicle_id_feasible)
        vehicle['unserved_customers'] = best_inserted_route_feasible
        vehicle_data = neighborhood_search_robust(vehicle_data, problem_data, distance_matrix, travel_time_disturbance, demand_disturbance, None, iteration_max, p_kr)
    else:
        vehicle = next(v for v in vehicle_data if v['vehicle_id'] == best_vehicle_id_infeasible)
        original_vehicle_data = copy.deepcopy(vehicle_data)  # 保存原来的车辆数据
        vehicle['unserved_customers'] = best_inserted_route_infeasible # 先把客户插到不可行位置
        # 邻域搜索
        vehicle_data = neighborhood_search_robust(vehicle_data, problem_data, distance_matrix, travel_time_disturbance, demand_disturbance, None, iteration_max, p_kr)
        if not check_feasibility_all_robust(vehicle_data, problem_data, distance_matrix, travel_time_disturbance, demand_disturbance):
            # 如果邻域搜索后仍然不可行，则恢复原来的车辆数据
            vehicle_data = original_vehicle_data
            # 创建一个新的车辆来服务新客户
            new_vehicle = {
                'vehicle_id': len(vehicle_data),
                'current_customer': 0,  # 当前服务的客户
                'status': 'serving',  # 对当前客户的车辆状态
                'expected_service_end_time': current_time,  # 当前客户预计服务结束时间
                'unserved_customers': [new_customer, 0],  # 新车辆的未服务客户，初始为新客户和仓库
                'served_customers': [0],
                'used_capacity': problem_data['DEMAND'][0],  # 新车辆的已用容量为新客户的需求
                'travel_data': [{
                    'customer': 0,
                    'arrive_time': current_time,  # 实际到达时间
                    'service_start_time': current_time,  # 开始服务时间
                    'service_end_time': current_time,  # 结束服务时间
                    'leave_time': current_time,  # 离开时间(当前模式是车辆会在服务完成后立即离开，到下一客户处等待)
                }],  # 记录实际的车辆行驶数据
            }
            vehicle_data.append(new_vehicle)
        else:
            pass
    return vehicle_data

def update_vehicle_data(vehicle_data, real_problem_data, real_travel_time, current_time):
    '''
    更新车辆数据
    :param vehicle_data: 车辆数据
    :param real_problem_data: 实际问题数据
    :param real_travel_time: 真实旅行时间矩阵
    :param current_time: 当前时间
    :return: 更新后的车辆数据
    '''
    # 车辆对于每个客户包含on_route, waiting_before, serving, waiting_after四个状态, 返回仓库后进入finished等状态
    # 只有车辆离开上一个客户后，才会更新travel_data，随后判断当前客户及对应状态
    for vehicle in vehicle_data:
        if vehicle['status'] == 'finished':
            continue

        unserved_customers = vehicle['unserved_customers'][:].copy()  # 复制未服务客户列表，避免在遍历时修改列表导致错误
        for customer in unserved_customers:
            # 获取上一个客户的旅行数据
            last_customer_travel_date = vehicle['travel_data'][-1]
            last_customer = last_customer_travel_date['customer']
            leave_last_customer_time = last_customer_travel_date['leave_time']

            # 计算应该到达当前客户的时间
            arrive_time = leave_last_customer_time + real_travel_time[last_customer, customer]
            service_start_time = max(arrive_time, real_problem_data['READY_TIME'][customer])
            service_end_time = service_start_time + real_problem_data['SERVICE_TIME'][customer]
            leave_time = service_end_time

            if current_time >= leave_time: # 如果当前时间大于等于离开时间，代表车辆已经完成服务当前客户
                vehicle['unserved_customers'].remove(customer)
                vehicle['served_customers'].append(customer)
                vehicle['used_capacity'] += real_problem_data['DEMAND'][customer]  # 更新车辆已用容量
                # 当前客户暂不更新
                # status暂不更新
                # 将当前客户的旅行数据添加到列表中
                vehicle['travel_data'].append({
                    'customer': customer,
                    'arrive_time': arrive_time,
                    'service_start_time': service_start_time,
                    'service_end_time': service_end_time,
                    'leave_time': leave_time
                })
                # print(f"车辆 {vehicle['vehicle_id']} 已经完成服务客户 {customer}， 离开时间: {leave_time}")
                if customer == 0:
                    vehicle['status'] = 'finished'
                continue  # 继续处理下一个未服务客户
            else: # 如果当前时间小于离开时间，代表车辆还在当前客户，进一步判断车辆状态
                vehicle['current_customer'] = customer  # 更新当前客户
                vehicle['expected_service_end_time'] = service_end_time  # 更新当前客户预计服务结束时间
                if current_time <= arrive_time:  # 如果当前时间小于等于到达时间，代表车辆还未到达当前客户
                    vehicle['status'] = 'on_route'
                elif current_time <= service_start_time:  # 如果当前时间小于等于服务开始时间，代表车辆还在等待服务
                    vehicle['status'] = 'waiting_before'
                elif current_time <= service_end_time:  # 如果当前时间小于等于服务结束时间，代表车辆正在服务当前客户
                    vehicle['status'] = 'serving'
                else:  # 如果当前时间大于服务结束时间，代表车辆已经完成服务当前客户
                    vehicle['status'] = 'waiting_after'
                # print(f"车辆 {vehicle['vehicle_id']} 当前状态: {vehicle['status']}, 当前客户: {customer}")
                break  # 跳出循环，等待下一个客户到达或状态更新
    return vehicle_data


def check_feasibility_robust(route, start_time, problem_data, travel_time_matrix, current_capacity_used, travel_time_disturbance=0, demand_disturbance=0):
    '''
    检查车辆路径的可行性
    :param route: 路径
    :param start_time: 起始时间
    :param problem_data: 问题数据
    :param travel_time_matrix: 旅行时间矩阵
    :param current_capacity_used: 当前已用容量
    :param delay_tolerance_time: 延误容忍时间
    :return: 是否可行
    '''
    capacity = problem_data['capacity']
    capacity_feasible = True
    time_window_feasible = True

    capacity_used = current_capacity_used
 
    if route:
        time = start_time
        last_customer = route[0]
        capacity_used += problem_data['DEMAND'][last_customer]
        delay_time = 0
        for customer in route[1:]:
            # 计算到达当前客户的时间
            arrive_time = max(time + travel_time_matrix[last_customer, customer], problem_data['READY_TIME'][customer]) + travel_time_disturbance
            # 计算延误时间
            due_date = problem_data['DUE_DATE'][customer]
            delay_time_customer = max(0, arrive_time - due_date)
            if delay_time_customer > 0:
                time_window_feasible = False
            if customer == 0:
                if delay_time_customer > 0:
                    time_window_feasible = False
            delay_time += delay_time_customer
            # 更新时间  
            time = arrive_time + problem_data['SERVICE_TIME'][customer]

            capacity_used += problem_data['DEMAND'][customer]

            last_customer = customer

    if capacity_used > capacity:
        capacity_feasible = False

    return capacity_feasible and time_window_feasible

def check_feasibility_all_robust(vehicle_data, problem_data, travel_time_matrix, travel_time_disturbance=0, demand_disturbance=0):
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

    for vehicle in vehicle_data:
        capacity_used = vehicle['used_capacity']  # 当前车辆已用容量

        if vehicle['unserved_customers']:
            route = vehicle['unserved_customers']
            leave_last_customer_time = vehicle['expected_service_end_time']
            time = leave_last_customer_time
            delay_time = 0
            last_customer = route[0]  
            capacity_used += problem_data['DEMAND'][last_customer]
            for customer in route[1:]:  # 跳过第一个客户（仓库）
                # 计算到达当前客户的时间
                arrive_time = max(time + travel_time_matrix[last_customer, customer], problem_data['READY_TIME'][customer]) + travel_time_disturbance
                # 计算延误时间
                due_date = problem_data['DUE_DATE'][customer]
                delay_time_customer = max(0, arrive_time - due_date)
                if delay_time_customer > 0:
                    time_window_feasible = False
                if customer == 0:
                    if delay_time_customer > 0:
                        time_window_feasible = False
                delay_time += delay_time_customer
                # 更新时间  
                time = arrive_time + problem_data['SERVICE_TIME'][customer]

                capacity_used += problem_data['DEMAND'][customer]

                last_customer = customer

        if capacity_used > capacity:
            capacity_feasible = False

    return capacity_feasible and time_window_feasible