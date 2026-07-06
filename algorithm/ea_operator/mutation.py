import random
import numpy as np
import sys
import os
from copy import deepcopy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..benchmark_process import calculate_distance, calculate_time

def mutate(individual):
    """
    对个体进行变异操作
    """
    # 选择变异操作
    mutation_type = random.choice(['swap', 'insert', 'reverse'])
    if mutation_type == 'swap':
        new_individual = swap_mutation(individual)
    elif mutation_type == 'insert':
        new_individual = insertion_mutation(individual)
    elif mutation_type == 'reverse':
        new_individual = inversion_mutation(individual)

    # 清除空路径
    new_individual = [route for route in new_individual if len(route) > 2]

    return new_individual

def swap_mutation(routes, inter_prob=0.3):
    """
    Swap mutation that can be intra-route or inter-route.
    inter_prob: probability to perform inter-route swap.
    """
    new_routes = deepcopy(routes)
    if random.random() < inter_prob and len(new_routes) >= 2:
        # inter-route swap
        r1, r2 = random.sample(range(len(new_routes)), 2)
        route1, route2 = new_routes[r1], new_routes[r2]
        if len(route1) > 2 and len(route2) > 2:
            i = random.randrange(1, len(route1)-1)
            j = random.randrange(1, len(route2)-1)
            route1[i], route2[j] = route2[j], route1[i]
    else:
        # intra-route swap
        r = random.randrange(len(new_routes))
        route = new_routes[r]
        if len(route) > 3:
            i, j = random.sample(range(1, len(route)-1), 2)
            route[i], route[j] = route[j], route[i]
    return new_routes

def insertion_mutation(routes, inter_prob=0.3):
    """
    Insertion mutation that can be intra-route or inter-route.
    inter_prob: probability to perform inter-route relocation.
    """
    new_routes = deepcopy(routes)
    if random.random() < inter_prob and len(new_routes) >= 2:
        # inter-route relocate
        from_r, to_r = random.sample(range(len(new_routes)), 2)
        route_from, route_to = new_routes[from_r], new_routes[to_r]
        if len(route_from) > 2:
            i = random.randrange(1, len(route_from)-1)
            customer = route_from.pop(i)
            j = random.randrange(1, len(route_to))
            route_to.insert(j, customer)
    else:
        # intra-route insertion
        r = random.randrange(len(new_routes))
        route = new_routes[r]
        if len(route) > 3:
            i, j = random.sample(range(1, len(route)-1), 2)
            customer = route.pop(i)
            route.insert(j, customer)
    return new_routes

# 逆转 Mutation（仅 intra-route）
def inversion_mutation(routes):
    """
    Inversion mutation (2-opt) applied within a single route.
    """
    new_routes = deepcopy(routes)
    r = random.randrange(len(new_routes))
    route = new_routes[r]
    if len(route) > 4:
        i, j = sorted(random.sample(range(1, len(route)-1), 2))
        route[i:j] = reversed(route[i:j])
    return new_routes