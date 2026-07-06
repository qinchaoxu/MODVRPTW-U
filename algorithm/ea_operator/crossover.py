'''
Description: 交叉操作
'''

import random

def path_oriented_crossover(parent1, parent2, child_count=2):
    """
    路径导向交叉(POX)：先解码父代得到路径，在路径级别进行交叉，再编码回巨型路径
    相比直接在巨型路径上交叉，这种方法更能保留有效的路径结构
    """

    routes1 = parent1
    routes2 = parent2
    
    # 随机选择一些路径保留给第一个子代
    preserve_ratio = random.uniform(0.3, 0.7)
    routes_to_preserve = max(1, int(len(routes1) * preserve_ratio))

    preserved_routes_idx = random.sample(range(len(routes1)), routes_to_preserve)

    # 从父代1中保留选定的路径
    child1_routes = [routes1[i] for i in preserved_routes_idx]
    
    # 记录已分配的客户
    assigned_customers = set()
    for route in child1_routes:
        for customer in route:
            if customer != 0:  # 不包括仓库
                assigned_customers.add(customer)
    
    # 从父代2中选择未分配的客户的路径
    for route in routes2:
        # 提取路径中未分配的客户
        unassigned = [c for c in route if c != 0 and c not in assigned_customers]
        if unassigned:
            # 创建新路径包含这些客户
            new_route = [0] + unassigned + [0]
            child1_routes.append(new_route)
            # 更新已分配客户集合
            assigned_customers.update(unassigned)
    
    if child_count == 1:
        return child1_routes, None

    # 对第二个子代重复相同的过程，但交换父代角色
    preserved_routes_idx = random.sample(range(len(routes2)), 
                                        max(1, int(len(routes2) * preserve_ratio)))
    child2_routes = [routes2[i] for i in preserved_routes_idx]
    
    assigned_customers = set()
    for route in child2_routes:
        for customer in route:
            if customer != 0:
                assigned_customers.add(customer)
    
    for route in routes1:
        unassigned = [c for c in route if c != 0 and c not in assigned_customers]
        if unassigned:
            new_route = [0] + unassigned + [0]
            child2_routes.append(new_route)
            assigned_customers.update(unassigned)
    
    return child1_routes, child2_routes