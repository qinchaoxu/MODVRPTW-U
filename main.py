"""
Main script for multi-objective vehicle routing with dynamic customers and robustness.
Features: Disturbances are pre-generated; each run uses the same disturbance data.
Usage: Testing algorithm performance with dynamic customer insertion and robustness evaluation.
"""

import numpy as np
import copy
import time
from algorithm.MOVRPTW_MOEAD_robust import MOVRPTW_MOEAD_Robust
from algorithm.benchmark_process import load_data, read_vehicle
from dynamic_customer import update_data, update_vehicle_data\
                            , insert_dynamic_customers_robust, generate_dynamic_customers_one, update_disturbed_data
import matplotlib.pyplot as plt
from robustness import robustness_check, add_stochastic_disturbance_all_customers

def main():

    problem_type = 'c1'  # Problem type, choose from: r1, r2, c1, c2, rc1, rc2
    if problem_type == 'r1':
        problem = 'r101'  # Problem name
    elif problem_type == 'r2':
        problem = 'r201'
    elif problem_type == 'c1':
        problem = 'c101'  # Problem name
    elif problem_type == 'c2':
        problem = 'c201'
    elif problem_type == 'rc1':
        problem = 'rc101'
    elif problem_type == 'rc2':
        problem = 'rc201'

    travel_time_disturbance = 1  # Travel time disturbance
    demand_disturbance = 3  # Demand disturbance
    dynamic_degree = 0.25


    print(f"Running trial with problem: {problem}, dynamic degree {dynamic_degree}, travel time disturbance: {travel_time_disturbance}, demand disturbance: {demand_disturbance}")
    customer_count = 100  # Number of customers

    data_path = f'data/solomon/{problem}.txt'
    data = load_data(data_path, customer_num=customer_count)

    # -------------- Generate dynamic customer data (randomly generated each run) ----------
    customer_data = generate_dynamic_customers_one(data, dynamic_customers_ratio=dynamic_degree)

    # ------------ Add disturbance data (randomly generated each run) ----------
    disturbed_customer_data, disturbed_travel_time = add_stochastic_disturbance_all_customers(customer_data, travel_time_disturbance, demand_disturbance)
    
    # ------------- Partition static and dynamic data ----------
    static_indices = []
    dynamic_indices = []
    for i in range(len(customer_data)):
        if customer_data.iloc[i]['ARRIVAL_TIME'] == 0:
            static_indices.append(i)
        else:
            dynamic_indices.append(i)
    dynamic_customers_data, static_customers_data = customer_data.iloc[dynamic_indices].copy(), customer_data.iloc[static_indices].copy()


    max_vehicle, capacity = read_vehicle(file_path=data_path)
    all_data = {
        'data': static_customers_data,
        'max_vehicle': max_vehicle,
        'capacity': capacity
    }

    # ---------- Initialize optimization algorithm ----------
    objectives = [0, 1, 4]

    movrptw_moead = MOVRPTW_MOEAD_Robust(
        data=all_data,
        population_size=50,
        max_generations=100,
        generation_opt=20,
        neighborhood_size=10,
        local_search_rate=1,
        objectives_index=objectives,  # Select objective function indices
        num_objectives=len(objectives),  # Number of objectives
        travel_time_disturbance=travel_time_disturbance,  # Travel time disturbance
        demand_disturbance=demand_disturbance,  # Demand disturbance
    )

    results = [[]]
    problem_data = copy.deepcopy(movrptw_moead.problem_data)
    distance_matrix = copy.deepcopy(movrptw_moead.distance_matrix)

    # --------- Solve ----------
    start_time = time.time()
    solutions, _ = movrptw_moead.run()
    end_time = time.time()
    static_time = end_time - start_time
    print(f"Algorithm run time: {end_time - start_time:.2f} seconds")

    # Plot Pareto front with all possible objective combinations
    all_objectives_names = ["vehicle_count", "total_distance", "total_wait_time", "total_delay_time", "std_redundancy_index"]
    objectives_names = [all_objectives_names[i] for i in movrptw_moead.objectives_index]

    results = [sol for sol in solutions if sol is not None]
    # Remove duplicates by objectives
    results = deduplicate_by_objectives(results)

    # Output best solutions for each objective
    best_solutions = []
    for i in range(movrptw_moead.num_objectives):
        best_solution = min(results, key=lambda x: x['objectives'][i])
        best_solutions.append(best_solution)
        print(f"Best {objectives_names[i]}: {best_solution['objectives']}")
        print(f"Corresponding route: {best_solution['solution']}")

    # Selected knee point solution
    selected_solution = movrptw_moead.decision_making(results)
    print(f"Selected solution: {selected_solution['objectives']}")
    print(f"Corresponding route: {selected_solution['solution']}")

    # ------------- Handle disturbances ----------
    arrived_customers = static_indices.copy()  # Dynamic customer indices
    disturbed_problem_data = copy.deepcopy(problem_data)  # In practical planning, real problem data may have changes in customer time windows and demands; here we assume using static original data
    # Use generated disturbance data instead of reading from file
    disturbed_customer_data_all = disturbed_customer_data
    disturbed_travel_time_all = disturbed_travel_time.tolist() if isinstance(disturbed_travel_time, np.ndarray) else disturbed_travel_time
    for i in range(len(static_indices)):
        customer_index = static_indices[i]
        disturbed_problem_data['DEMAND'][i] = disturbed_customer_data_all.iloc[customer_index]['DEMAND']

    disturbed_travel_time = np.array([[disturbed_travel_time_all[i][j] for j in static_indices] for i in static_indices])  # Keep only distances between static customers

    # ------------ Dynamic routing part ----------
    vehicle_data = []
    # Generate vehicle travel data
    for i, route in enumerate(selected_solution['solution']):
        vehicle_data.append({
            'vehicle_id': i,
            'current_customer': 0,  # Current customer being served
            'status': 'serving',  # Vehicle status for current customer
            'expected_service_end_time': 0,  # Expected service end time for current customer
            'unserved_customers': route[1:],
            'served_customers': [0],
            'used_capacity': problem_data['DEMAND'][0],  # Initial vehicle capacity (depot demand)
            'travel_data': [{
                'customer': 0,
                'arrive_time': 0,  # Actual arrival time
                'service_start_time': 0,  # Service start time
                'service_end_time': 0,  # Service end time
                'leave_time': 0,  # Leave time (vehicle leaves after service to next customer location)
            }],  # Record actual vehicle travel data
        })

    # Sort dynamic data by arrival time
    dynamic_customers_data = dynamic_customers_data.sort_values(by='ARRIVAL_TIME').reset_index(drop=True)

    # Simulate dynamic customer arrival
    start_time = time.time()
    dynamic_customer_indices = []
    static_customer_num = len(static_indices)
    for index, row in dynamic_customers_data.iterrows():
        print(f"Dynamic customer {index+1}/{len(dynamic_customers_data)} arrived, customer ID: {int(row['CUST_NO'])}, arrival time: {row['ARRIVAL_TIME']}")
        current_time = row['ARRIVAL_TIME']
        vehicle_data = update_vehicle_data(vehicle_data, disturbed_problem_data, disturbed_travel_time, current_time)
        # Update data
        dynamic_customer = copy.deepcopy(row)
        problem_data, distance_matrix = update_data(problem_data, distance_matrix, row)

        arrived_customers.append(int(dynamic_customer['CUST_NO']))
        dynamic_customer_indices.append(static_customer_num + index)
        disturbed_problem_data, disturbed_travel_time = update_disturbed_data(disturbed_problem_data, dynamic_customer, disturbed_customer_data_all, disturbed_travel_time_all, arrived_customers)


        vehicle_data = insert_dynamic_customers_robust(vehicle_data, row, current_time
                                                , problem_data, distance_matrix, 0
                                                , disturbed_problem_data, travel_time_disturbance, demand_disturbance, v=1, iteration_max=100, p_kr=0.3)

    vehicle_data = update_vehicle_data(vehicle_data, disturbed_problem_data, disturbed_travel_time, 5000)
    end_time = time.time()
    total_delay_time, total_capacity_violation = robustness_check(vehicle_data, disturbed_problem_data, disturbed_travel_time)
    distance = cal_total_distance(vehicle_data, distance_matrix)
    total_distance = sum(vehicle['total_distance'] for vehicle in distance)

    print(f"Vehicle count: {len(vehicle_data)}, Total distance: {total_distance}, Total delay time: {total_delay_time}, Total capacity violation: {total_capacity_violation}")

def deduplicate_by_objectives(results):
    """
    Remove duplicates by objective values (ignoring solution and other fields).
    """
    seen = set()
    unique_results = []
    for d in results:
        # Convert numpy.ndarray to tuple as hash key
        obj_key = tuple(np.round(d['objectives'], decimals=8).tolist())  # Optional: round to prevent floating point errors
        if obj_key not in seen:
            seen.add(obj_key)
            unique_results.append(d)
    return unique_results

def cal_total_distance(vehicle_data, distance_matrix):
    '''
    Calculate total distance of vehicle routes
    :param vehicle_data: Vehicle data
    :param distance_matrix: Distance matrix
    :return: Total distance
    '''
    distance_routes = []
    for vehicle in vehicle_data:
        total_distance = 0
        if vehicle['served_customers']:
            for i in range(1, len(vehicle['served_customers'])):
                total_distance += distance_matrix[vehicle['served_customers'][i - 1], vehicle['served_customers'][i]]
            # Return to depot distance
            total_distance += distance_matrix[vehicle['served_customers'][-1], 0]  # Assume depot ID is 0
        distance_routes.append({
            'vehicle_id': vehicle['vehicle_id'],
            'total_distance': total_distance
        })

    return distance_routes


def save_routes(vehicle_data, disturbed_problem_data, disturbed_travel_time, dynamic_customer_indices, file_path):
    """
    Save vehicle routes, problem_data and travel_time_matrix to JSON file
    """
    import json
    routes = [vehicle['served_customers'] for vehicle in vehicle_data]
    # cust_no = disturbed_problem_data['CUST_NO'].tolist() + dynamic_customer_indices
    # for route in routes:
    #     for i in range(len(route)):
    #         route[i] = int(cust_no[route[i]])
    for k, v in disturbed_problem_data.items():
        if isinstance(v, np.ndarray):
            disturbed_problem_data[k] = v.tolist()  # Convert to list
    disturbed_travel_time = disturbed_travel_time.tolist()
    data_to_save = {
        'routes': routes,
        'problem_data': disturbed_problem_data,
        'travel_time_matrix': disturbed_travel_time,
        'dynamic_customer_indices': dynamic_customer_indices
    }
    with open(file_path, 'w') as f:
        json.dump(data_to_save, f)


if __name__ == "__main__":
    main()
