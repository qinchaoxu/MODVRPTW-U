import numpy as np
import random
import math
import copy
import itertools
import sklearn
from typing import List, Tuple, Dict, Any, Callable
from algorithm.cal_objective import cal_vehicle_num, cal_stdi_obj, cal_total_distance_and_wait_time_and_delay_time
from algorithm.benchmark_process import load_data, cal_distance_matrix, read_vehicle
from algorithm.ea_operator.initialization import random_initialization, objective_wise_initialization
import algorithm.ea_operator.crossover as crossover
import algorithm.ea_operator.mutation as mutation
from algorithm.ea_operator.local_search_optimal import objective_wise_local_search
from algorithm.ea_operator.local_search_robust import robustness_focused_local_search
from algorithm.robustness_calculation import robustness_test_routes

class MOVRPTW_MOEAD_Robust:
    """
    多目标进化算法/分解 (MOEA/D) 求解多目标车辆路径问题带时间窗 (MOVRPTW)
    """
    
    def __init__(self, 
                 data: Dict[str, Any],
                 distance_matrix: np.ndarray = None,
                 travel_time_matrix: np.ndarray = None,
                 encoder: Callable = None,
                 decoder: Callable = None,
                 population_size: int = 100,
                 max_generations: int = 500,
                 generation_opt: int = 0,
                 neighborhood_size: int = 20,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.1,
                 local_search_rate: float = 0.5,
                 weight_vectors: np.ndarray = None,
                 num_objectives: int = 5,
                 objectives_index: List[int] = None,
                 feasibility_check_item: List[str] = None,
                 theta: float = 1.0,
                 travel_time_disturbance: float = 3.0,
                 demand_disturbance: float = 3.0,
                 test_times: int = 50):
        """
        初始化MOEA/D算法

        参数:
            problem_data: benchmark_process处理后的数据，包含VRPTW问题的所有数据，例如客户节点、车辆容量、时间窗等
            distance_matrix: 预先计算的距离矩阵，如果为None则自动计算
            encoder: 编码函数，将路径表示转换为染色体表示，默认使用路径表示
            decoder: 解码函数，将染色体表示转换为路径表示，默认使用路径表示
            population_size: 种群大小
            max_generations: 最大迭代次数
            generation_opt: 开始进行鲁棒性增强的迭代次数，如果为0则在最后一半迭代进行鲁棒性测试
            neighborhood_size: 每个权重向量的邻域大小
            crossover_rate: 交叉率
            mutation_rate: 变异率
            local_search_rate: 局部搜索率
            weight_vectors: 预定义的权重向量，如果为None则自动生成
            num_objectives: 目标函数数量
            objectives_index: 目标函数索引，如果为None则自动选择所有目标（0：车辆数，1：总成本，2：总行驶时间，3：总延迟时间，4：冗余空间均匀度指标）
            feasibility_check_item: 可行性检查项，如果为None则检查所有项（"vehicle_num", "capacity", "time_window", "customer_visit"）
            theta: PBI距离中的参数，控制投影距离和垂直距离的权重
            travel_time_disturbance: 旅行时间扰动，用于鲁棒性测试
            demand_disturbance: 需求扰动，用于鲁棒性测试
            test_times: 测试次数，用于鲁棒性测试
        """

        # 解析数据
        self.customer_no = data['data']['CUST_NO'].to_numpy()
        self.ready_time = data['data']['READY_TIME'].to_numpy()
        self.service_time = data['data']['SERVICE_TIME'].to_numpy()
        self.due_date = data['data']['DUE_DATE'].to_numpy()
        self.demand = data['data']['DEMAND'].to_numpy()
        self.x_coord = data['data']['XCOORD'].to_numpy()
        self.y_coord = data['data']['YCOORD'].to_numpy()

        # 检查是否包含max_vehicle
        if 'max_vehicle' not in data:
            # 设置为None
            data['max_vehicle'] = None
        # 检查是否包含capacity
        if 'capacity' not in data:
            # 抛出错误
            raise ValueError("数据中缺少 'capacity' 字段，请确保输入数据包含车辆容量信息")
        

        self.problem_data = {
            'CUST_NO': self.customer_no,
            'READY_TIME': self.ready_time,
            'SERVICE_TIME': self.service_time,
            'DUE_DATE': self.due_date,
            'DEMAND': self.demand,
            'XCOORD': self.x_coord,
            'YCOORD': self.y_coord,
            'max_vehicle': data['max_vehicle'],
            'capacity': data['capacity']
        }

        self.encoder = encoder
        self.decoder = decoder
        self.population_size = population_size
        self.max_generations = max_generations
        if generation_opt == 0:
            self.generation_opt = max_generations // 2
        else:
            self.generation_opt = generation_opt
        self.neighborhood_size = neighborhood_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.local_search_rate = local_search_rate
        self.num_objectives = num_objectives
        self.theta = theta
        if objectives_index is None:
            # 根据目标数量自动选择目标
            self.objectives_index = list(range(num_objectives))
        else:
            # 判断目标数量是否一致
            if len(objectives_index) != num_objectives:
                raise ValueError("目标数量与目标索引数量不一致")
            self.objectives_index = objectives_index
        
        if feasibility_check_item is None and (3 in self.objectives_index or 4 in self.objectives_index):
            # 目标包含延迟时间或者均匀指数，默认为软时间窗，不检查时间窗可行性
            self.feasibility_check_item = ["vehicle_num", "capacity", "time_window_soft", "customer_visit"]
        elif feasibility_check_item is None:
            if self.problem_data['max_vehicle'] is None:
                # 如果没有max_vehicle字段，则不检查车辆数量约束
                self.feasibility_check_item = ["capacity", "time_window", "customer_visit"]
            else:
                self.feasibility_check_item = ["vehicle_num", "capacity", "time_window", "customer_visit"]
        else:
            # 检查约束项是否有效
            valid_check_items = ["vehicle_num", "capacity", "time_window", "customer_visit"]
            for item in feasibility_check_item:
                if item not in valid_check_items:
                    raise ValueError(f"无效的可行性检查项: {item}")
            self.feasibility_check_item = feasibility_check_item

        # 鲁棒性测试相关参数
        self.travel_time_disturbance = travel_time_disturbance
        self.demand_disturbance = demand_disturbance
        self.test_times = test_times

        # 计算距离矩阵
        if distance_matrix is None:
            self.distance_matrix = cal_distance_matrix(self.problem_data)
        else:
            self.distance_matrix = distance_matrix
        # 如果提供了旅行时间矩阵，则使用它，否则旅行时间默认等于距离
        if travel_time_matrix is None:
            self.travel_time_matrix = self.distance_matrix
        else:
            self.travel_time_matrix = travel_time_matrix
        self.customer_count = len(self.demand) - 1   # 不包括仓库
        self.max_vehicle, self.capacity = data['max_vehicle'], data['capacity']
        service_times = data['data']['SERVICE_TIME'].values
        self.mean_service_time = np.mean(service_times)
        self.mean_travel_time = np.mean(self.travel_time_matrix)
        self.high_redundancy_threshold = self.mean_travel_time + self.mean_service_time  # 冗余空间均匀度指标的阈值
        
        # 生成或使用预定义的权重向量
        if weight_vectors is None:
            self.weight_vectors = self._generate_weight_vectors()
        else:
            self.weight_vectors = weight_vectors
            
        # 确定种群大小与权重向量数量一致
        self.population_size = len(self.weight_vectors)
        
        # 计算权重向量之间的欧几里得距离并确定邻域
        self.neighborhood = self._compute_neighborhood()
        
        # 种群，每个解是一个路径表示
        self.population = []
        
        # 理想点，用于计算切比雪夫距离
        self.ideal_point = np.array([float('inf')] * self.num_objectives)
        self.nadir_point = np.array([float('-inf')] * self.num_objectives)

        # 初始化外部鲁棒性解集
        self.external_robust_archive = []

        # # 初始化外部鲁棒性解集，大小等于种群大小
        # self.robust_population = [None] * self.population_size

    def _generate_weight_vectors(self) -> np.ndarray:
        """生成均匀分布的权重向量，支持不同的目标数量
        
        返回:
            均匀分布的权重向量数组
        """
        return self._generate_systematic_weight_vectors(self.num_objectives, self.population_size)
    
    def _generate_systematic_weight_vectors(self, m: int, n: int) -> np.ndarray:
        """
        使用系统生成法生成权重向量，适用于任意目标数量
        
        参数:
            m: 目标函数个数
            n: 需要的权重向量数量
            
        返回:
            均匀分布的权重向量数组
        """
        # 基于整数格子法（Das and Dennis方法）
        # 估算适当的划分层数h，使得生成的向量数量接近n
        h = 1
        while math.comb(h+m-1, m-1) < n:
            h += 1
        
        # 如果h过大（可能导致生成太多向量），则回退到h-1
        if h > 1 and math.comb(h+m-1, m-1) > 2*n:
            h -= 1
        
        # 生成所有可能的m维向量，其元素和为h
        weight_vectors = []
        
        def generate_weights(current_vector, remaining, index):
            if index == m - 1:
                # 最后一个元素直接由剩余值确定
                new_vector = current_vector.copy()
                new_vector[index] = remaining
                weight_vectors.append(new_vector)
                return
            
            for i in range(remaining + 1):
                new_vector = current_vector.copy()
                new_vector[index] = i
                generate_weights(new_vector, remaining - i, index + 1)
        
        # 开始递归生成
        generate_weights([0] * m, h, 0)
        
        # 将整数权重转换为[0,1]范围内的实数
        weight_vectors = np.array(weight_vectors) / h
        
        # 添加单位向量（只考虑单个目标的极端情况）
        boundary_points = np.eye(m)
        weight_vectors = np.vstack([weight_vectors, boundary_points])
        
        # 如果生成的向量数量不等于n，选择最具代表性的n个向量
        if len(weight_vectors) != n:
            if len(weight_vectors) > n:
                # 如果生成了太多向量，使用k-means聚类选择具有代表性的子集
                try:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n, random_state=0).fit(weight_vectors)
                    centers = kmeans.cluster_centers_
                    # 归一化聚类中心，确保权重和为1
                    centers = centers / np.sum(centers, axis=1, keepdims=True)
                    return centers
                except ImportError:
                    # 如果没有sklearn，则使用均匀抽样
                    indices = np.linspace(0, len(weight_vectors)-1, n, dtype=int)
                    return weight_vectors[indices]
            else:
                # 如果生成的向量太少，随机生成额外的向量
                additional_vectors = self._generate_random_weight_vectors(n - len(weight_vectors))
                return np.vstack([weight_vectors, additional_vectors])
        
        return weight_vectors
    
    def _generate_random_weight_vectors(self, n: int) -> np.ndarray:
        """生成n个随机权重向量"""
        vectors = []
        for _ in range(n):
            w = np.random.random(self.num_objectives)
            vectors.append(w / np.sum(w))  # 归一化确保权重和为1
        return np.array(vectors)
    
    def _compute_neighborhood(self) -> List[List[int]]:
        """计算每个权重向量的邻域"""
        distance_matrix = np.zeros((self.population_size, self.population_size))
        neighborhood = []
        
        # 计算每对权重向量之间的欧几里得距离
        for i in range(self.population_size):
            for j in range(self.population_size):
                distance_matrix[i, j] = np.linalg.norm(
                    self.weight_vectors[i] - self.weight_vectors[j]
                )
        
        # 对于每个权重向量，选择最近的邻居
        for i in range(self.population_size):
            sorted_indices = np.argsort(distance_matrix[i])
            neighborhood.append(sorted_indices[:self.neighborhood_size].tolist())
        
        return neighborhood
    
    def initialize_population(self, heuristic_ratio=0.2):
        """初始化种群，保证解–权重一一对应，并在初始化时进行修复"""
        self.population = []
        self.robust_population = []
        for i, weight in enumerate(self.weight_vectors):
            # === 修改1: 针对第 i 个权重向量单独生成一个解，保持一一对应
            if random.random() < heuristic_ratio:
                # solution = heuristic_initialization(
                #     self.problem_data, self.distance_matrix,
                #     self.customer_count, self.capacity
                # )
                # Select objective from those other than the 0-th objective
                objective_chosen = random.choice(self.objectives_index[1:])
                # print(f"Using objective-wise initialization, selected objective: {objective_chosen}")
                solution = objective_wise_initialization(
                    self.problem_data, self.distance_matrix, self.travel_time_matrix,
                    self.customer_count, self.capacity,
                    objective_chosen, self.high_redundancy_threshold, 
                    self.feasibility_check_item
                )
            else:
                # print("Using random initialization")
                solution = random_initialization(
                    self.max_vehicle, self.customer_count
                )

            individual, objectives, feasibility = self._evaluate_solution(solution, get_objectives_feasibility=True)
            self.population.append(individual)

            total_delay_time, total_capacity_violation = robustness_test_routes(
                individual['solution'], self.problem_data, self.travel_time_matrix,
                self.travel_time_disturbance, self.demand_disturbance, self.test_times
            )

            robustness_tested_solution = {
                'solution': copy.deepcopy(individual['solution']),
                'objectives': copy.deepcopy(individual['objectives']),
                'total_delay_time': total_delay_time,
                'total_capacity_violation': total_capacity_violation
            }
            self.robust_population.append(robustness_tested_solution)  # 初始化鲁棒性解集

            # === 修改3: 初始化时更新理想点
            self._update_ideal_point(objectives)
            self._update_nadir_point(objectives)

    def _evaluate_solution(self, solution, get_objectives_feasibility=False) -> Dict[str, Any]:
        """评估解的目标值和可行性"""
        # 计算目标值
        objectives = self._evaluate_objectives(solution)    
        # 检查可行性
        feasibility = self._check_feasibility(solution, check_item=self.feasibility_check_item)
        individual = {
            'solution': solution,
            'objectives': objectives,
            'feasibility': feasibility
        }
        # 如果需要返回目标值和可行性
        if get_objectives_feasibility:
            return individual, objectives, feasibility
        # 否则只返回个体
        return individual

    def _evaluate_objectives(self, solution: List[List[int]], objectives_index: List[int] = None) -> np.ndarray:
        """
        评估解的多个目标值
        
            参数:
                solution: 路径列表表示的解
                objectives_index: 目标函数索引，如果为None则自动选择所有目标（0：车辆数，2：总成本，3：总行驶时间，4：总延迟时间，5：冗余空间均匀度指标）

        返回:
            包含各目标值的NumPy数组
        """
        if self.decoder is not None:
            solution = self.decoder(solution) # 如果解码器不为空，则解码

        vehicle_num = cal_vehicle_num(solution)
        total_cost, total_travel_time, total_delay_time = cal_total_distance_and_wait_time_and_delay_time(self.problem_data, self.distance_matrix, self.travel_time_matrix, solution)
        stdi_obj = cal_stdi_obj(self.problem_data, self.travel_time_matrix, solution)

        if objectives_index is None:
            objectives_index = self.objectives_index
        objectives = np.array([vehicle_num, total_cost, total_travel_time, total_delay_time, stdi_obj])
        objectives = objectives[objectives_index]

        return objectives
    
    def _check_feasibility(self, solution: List[List[int]], check_item: list[str] = None) -> Dict[str, bool]:
        """
        检查解的可行性
        
        参数:
            solution: 路径列表表示的解
            check_item: 检查项，如果为None则检查所有项,["vehicle_num", "capacity", "time_window", "customer_visit"]

        返回:
            包含各约束可行性的字典（不检查时返回True）
        """
        if not solution:
            return {"vehicle_num": True, "capacity": True, "time_window": True, "customer_visit": False, "feasible": False}
        
        if check_item is None:
            check_item = ["vehicle_num", "capacity", "time_window", "customer_visit"]
        
        vehicle_num_feasible = True
        # 检查车辆数量约束
        if "vehicle_num" in check_item:
            vehicle_count = sum(1 for route in solution if len(route) > 2)  # 路径长度大于2表示有客户
            if vehicle_count > self.max_vehicle:
                vehicle_num_feasible = False

        customer_visit_feasible = True
        # 检查客户访问唯一性
        if "customer_visit" in check_item:
            all_customers = set(range(1, self.customer_count + 1))
            visited_customers = set()
            for route in solution:
                for customer in route:
                    if customer != 0:  # 排除仓库
                        visited_customers.add(customer)
        
            customer_visit_feasible = visited_customers == all_customers
        
        # 检查容量约束
        capacity_feasible = True
        if "capacity" in check_item:
            for route in solution:
                if len(route) <= 2:  # 路径为空或只有仓库
                    continue
                
                # 计算路径上的总需求
                total_demand = sum(self.demand[c] for c in route if c != 0)
                
                # 如果总需求超过容量，标记为不可行
                if total_demand > self.capacity:
                    capacity_feasible = False
                    break
        
        # 检查时间窗约束
        time_window_feasible = True
        if "time_window" in check_item:
            for route in solution:
                if len(route) <= 2:  # 路径为空或只有仓库
                    continue
                
                # 模拟路径行驶过程
                current_time = 0  # 从时间0开始
                
                for i in range(1, len(route)):
                    prev_customer = route[i-1]
                    current_customer = route[i]
                    
                    # 计算旅行时间
                    travel_time = self.travel_time_matrix[prev_customer, current_customer]

                    # 更新当前时间
                    current_time += travel_time
                    
                    # 如果当前节点是客户（非仓库）
                    if current_customer != 0:
                        # 获取时间窗
                        ready_time = self.ready_time[current_customer]
                        due_time = self.due_date[current_customer]
                        service_time = self.service_time[current_customer]
                        
                        # 如果到达太早，等待
                        if current_time < ready_time:
                            current_time = ready_time
                        
                        # 如果到达太晚，标记为不可行
                        if current_time > due_time:
                            time_window_feasible = False
                            break
                        
                        # 添加服务时间
                        current_time += service_time
                
                if not time_window_feasible:
                    break

        if "time_window_soft" in check_item:
            for route in solution:
                if len(route) <= 2:  # 路径为空或只有仓库
                    continue
                
                # 模拟路径行驶过程
                current_time = 0  # 从时间0开始
                
                for i in range(1, len(route)):
                    prev_customer = route[i-1]
                    current_customer = route[i]
                    
                    # 计算旅行时间
                    travel_time = self.travel_time_matrix[prev_customer, current_customer]

                    # 更新当前时间
                    current_time += travel_time
                    
                    # 如果当前节点是客户（非仓库）
                    if current_customer != 0:
                        # 获取时间窗
                        ready_time = self.ready_time[current_customer]
                        due_time = self.due_date[current_customer]
                        service_time = self.service_time[current_customer]
                        
                        # 如果到达太早，等待
                        if current_time < ready_time:
                            current_time = ready_time
                        
                        # 如果到达太晚，标记为不可行
                        if current_time > due_time  + 0 * self.high_redundancy_threshold:
                            time_window_feasible = False
                            break
                        
                        # 添加服务时间
                        current_time += service_time
                
                if not time_window_feasible:
                    break
        
        return {
            "vehicle_num": vehicle_num_feasible,
            "capacity": capacity_feasible,
            "time_window": time_window_feasible,
            "customer_visit": customer_visit_feasible,
            "feasible": vehicle_num_feasible and capacity_feasible and time_window_feasible and customer_visit_feasible
        }
    
    def _update_ideal_point(self, objectives: np.ndarray):
        """更新理想点"""
        self.ideal_point = np.minimum(self.ideal_point, objectives)

    def _update_nadir_point(self, objectives: np.ndarray):
        """更新极大点"""
        self.nadir_point = np.maximum(self.nadir_point, objectives)
    
    def _tchebycheff_distance(self, objectives: np.ndarray, weight: np.ndarray) -> float:
        """计算给定解的切比雪夫距离"""
        # 先做归一化
        range_vec = self.nadir_point - self.ideal_point
        # 避免除以 0
        range_vec[range_vec == 0] = 1e-12
        norm_diff = (objectives - self.ideal_point) / range_vec
        # 计算切比雪夫距离
        tchebycheff_distance = np.max(weight * np.abs(norm_diff))
        return tchebycheff_distance
    
    def _pbi_distance(self, objectives: np.ndarray, weight: np.ndarray) -> float:
        """
        PBI 标量化距离：d1 是投影距离，d2 是垂直距离
        distance = d1 + theta * d2
        """
        # 先做归一化
        range_vec = self.nadir_point - self.ideal_point
        # 避免除以 0
        range_vec[range_vec == 0] = 1e-12
        norm_diff = (objectives - self.ideal_point) / range_vec

        norm_diff = objectives - self.ideal_point

        # 按 PBI 公式计算
        w_norm = np.linalg.norm(weight)
        d1 = np.dot(norm_diff, weight) / w_norm
        proj = (d1 / w_norm) * weight
        d2 = np.linalg.norm(norm_diff - proj)
        return d1 + self.theta * d2
    
    def _crossover(self, parent1: List[List[int]], parent2: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """执行两个父解之间的交叉操作"""
        return crossover.path_oriented_crossover(parent1, parent2, 1)

    def _mutation(self, solution: List[List[int]]) -> List[List[int]]:
        """对解进行变异操作"""
        return mutation.mutate(solution)
    
    def _local_search(self, solution: List[List[int]], weight_vector: List[float]) -> List[List[int]]:
        """对解进行局部搜索操作"""
        if self.generation >= self.generation_opt:
            total_delay_time, total_capacity_violation = robustness_test_routes(
                solution, self.problem_data, self.travel_time_matrix,
                self.travel_time_disturbance, self.demand_disturbance, self.test_times
                )
            if total_delay_time + total_capacity_violation == 0:
                # If current solution's robustness metric is 0, use objective-wise local search
                # print("Current solution is robust, using objective-wise local search")
                return objective_wise_local_search(solution, self.problem_data, self.distance_matrix, self.travel_time_matrix, self.max_vehicle, self.capacity, self.high_redundancy_threshold,
                                                weight_vector, self.objectives_index, self.feasibility_check_item)
            else:
                # print("Current solution is not robust, using robustness-focused local search")
                return robustness_focused_local_search(solution, self.problem_data, self.distance_matrix, self.travel_time_matrix, self.travel_time_disturbance, self.demand_disturbance)
        else:
            # 在前半段迭代中，使用目标导向的局部搜索
            return objective_wise_local_search(solution, self.problem_data, self.distance_matrix, self.travel_time_matrix, self.max_vehicle, self.capacity, self.high_redundancy_threshold,
                                            weight_vector, self.objectives_index, self.feasibility_check_item)

    def update_population(self, offspring_idx: int, offspring: Dict) -> int:
        """
        更新种群

        参数:
            offspring_idx: 新解的索引
            offspring: 新解
        """
        PENALTY = 1e3

        for i in self.neighborhood[offspring_idx]:
            old_objs = self.population[i]['objectives'].copy()
            new_objs = offspring['objectives'].copy()

            if not offspring['feasibility']['feasible']:
                new_objs += PENALTY
            if not self.population[i]['feasibility']['feasible']:
                old_objs += PENALTY

            weight = self.weight_vectors[i]
            old_fit = self._pbi_distance(old_objs, weight)
            new_fit = self._pbi_distance(new_objs, weight)

            if new_fit <= old_fit:
                self.population[i] = copy.deepcopy(offspring)
                self._update_ideal_point(offspring['objectives'])
                self._update_nadir_point(offspring['objectives'])

    def update_population_robust(self, offspring_idx: int, new_robustness_solution: Dict) -> int:
        """
        更新种群

        参数:
            offspring_idx: 新解的索引
            offspring: 新解
        """
        # 计算当前解的鲁棒性指标，越小越好
        robustness_indicator = new_robustness_solution['total_delay_time'] + new_robustness_solution['total_capacity_violation']
        new_objectives = np.array(new_robustness_solution['objectives'])
        updated = False

        for i in self.neighborhood[offspring_idx]:
            # 判断鲁棒性
            existing_robustness_indicator = self.population[i]['total_delay_time'] + \
                                self.population[i]['total_capacity_violation']
            if robustness_indicator < existing_robustness_indicator:
                # 如果新解的鲁棒性指标更好，无论如何都更新
                self.population[i] = copy.deepcopy(new_robustness_solution)
                updated = True
            elif robustness_indicator == existing_robustness_indicator:
                # 如果鲁棒性指标相同，比较PBI距离
                existing_objectives = np.array(self.population[i]['objectives'])
                if self._pbi_distance(new_objectives, self.weight_vectors[i]) < \
                self._pbi_distance(existing_objectives, self.weight_vectors[i]):
                    # 如果新解的PBI距离更小，更新
                    self.population[i] = copy.deepcopy(new_robustness_solution)
                    updated = True
        if updated:
            # 更新理想点和极大点
            self._update_ideal_point(new_objectives)
            self._update_nadir_point(new_objectives)

    def update_robust_population(self, offspring_idx: int, new_robustness_solution: Dict) -> int:
        """
        更新种群

        参数:
            offspring_idx: 新解的索引
            offspring: 新解
        """
        # 计算当前解的鲁棒性指标，越小越好
        robustness_indicator = new_robustness_solution['total_delay_time'] + new_robustness_solution['total_capacity_violation']
        new_objectives = np.array(new_robustness_solution['objectives'])
        updated = False

        for i in self.neighborhood[offspring_idx]:
            # 如果当前索引的外部鲁棒性解为空或新解鲁棒性更好
            if self.robust_population[i] is None:
                self.robust_population[i] = copy.deepcopy(new_robustness_solution)
                updated = True
            else:
                # 判断鲁棒性
                existing_robustness_indicator = self.robust_population[i]['total_delay_time'] + \
                                    self.robust_population[i]['total_capacity_violation']
                if robustness_indicator < existing_robustness_indicator:
                    # 如果新解的鲁棒性指标更好，无论如何都更新
                    self.robust_population[i] = copy.deepcopy(new_robustness_solution)
                    updated = True
                elif robustness_indicator == existing_robustness_indicator:
                    # 如果鲁棒性指标相同，比较PBI距离
                    existing_objectives = np.array(self.robust_population[i]['objectives'])
                    if self._pbi_distance(new_objectives, self.weight_vectors[i]) < \
                    self._pbi_distance(existing_objectives, self.weight_vectors[i]):
                        # 如果新解的PBI距离更小，更新
                        self.robust_population[i] = copy.deepcopy(new_robustness_solution)
                        updated = True
        if updated:
            # 更新理想点和极大点
            self._update_ideal_point(new_objectives)
            self._update_nadir_point(new_objectives)

    def update_external_archive(self, new_robustness_solution) -> None:
        """
        更新外部鲁棒性档案
        参数:
            new_robustness_solution: 新的待评估的鲁棒性解
        """
        # 计算当前解的鲁棒性指标
        robustness_indicator = new_robustness_solution['total_delay_time'] + new_robustness_solution['total_capacity_violation']

        if robustness_indicator == 0:
            # 如果鲁棒性指标为0，才有可能添加到外部鲁棒性解集，进行一次非支配档案更新
            is_dominated = False
            new_archive = []
            obj_new = new_robustness_solution['objectives']
            for sol in self.external_robust_archive:
                obj_sol = sol['objectives']
                if (np.all(obj_sol <= obj_new) and np.any(obj_sol < obj_new)):
                    # 如果现有解支配新解，则丢弃新解
                    is_dominated = True
                    break
                if not (np.all(obj_new <= obj_sol) and np.any(obj_new < obj_sol)):
                    # 如果新解不支配现有解，则保留现有解
                    new_archive.append(sol)

            if not is_dominated:
                # 如果新解不被支配，则添加到外部鲁棒性解集
                new_archive.append(new_robustness_solution)
                self.external_robust_archive = new_archive

    def _non_dominated_sort(self, solutions: List[Dict]) -> List[Dict]:
        """执行非支配排序，返回所有非支配解"""
        non_dominated = []
        
        for i, sol_i in enumerate(solutions):
            is_dominated = False
            
            for j, sol_j in enumerate(solutions):
                if i == j:
                    continue
                
                # 检查sol_j是否支配sol_i，考虑约束可行性
                if self._dominates(sol_j['objectives'], sol_i['objectives'], 
                                  sol_j.get('solution'), sol_i.get('solution'),
                                  sol_j['feasibility'], sol_i['feasibility']):
                    is_dominated = True
                    break
            
            if not is_dominated:
                non_dominated.append(copy.deepcopy(sol_i))
        
        return non_dominated
    
    def _non_dominated_sort_robust(self, solutions: List[Dict]) -> List[Dict]:
        """执行非支配排序，返回所有非支配解"""
        non_dominated = []
        
        for i, sol_i in enumerate(solutions):
            is_dominated = False
            sol_i['feasibility'] = {'feasible': True}
            
            for j, sol_j in enumerate(solutions):
                if i == j:
                    continue
                sol_j['feasibility'] = {'feasible': True}
                
                # 检查sol_j是否支配sol_i，考虑约束可行性
                if self._dominates(sol_j['objectives'], sol_i['objectives'], 
                                  sol_j.get('solution'), sol_i.get('solution'),
                                  sol_j['feasibility'], sol_i['feasibility']):
                    is_dominated = True
                    break
            
            if not is_dominated:
                non_dominated.append(copy.deepcopy(sol_i))
        
        return non_dominated
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray, sol1: List[List[int]] = None, sol2: List[List[int]] = None,
                   feasibility1: Dict[str, bool] = None, feasibility2: Dict[str, bool] = None) -> bool:
        """
        检查obj1是否支配obj2，考虑约束可行性
        
        参数:
            obj1: 第一个解的目标值
            obj2: 第二个解的目标值
            sol1: 第一个解（用于检查可行性）
            sol2: 第二个解（用于检查可行性）
            feasibility1: 第一个解的可行性
            feasibility2: 第二个解的可行性
        返回:
            是否支配
        """
        # 如果没有提供解，则只考虑目标值
        if sol1 is None or sol2 is None:
            # 假设所有目标都是最小化
            return (np.all(obj1 <= obj2) and np.any(obj1 < obj2))
          
        # 可行解总是支配不可行解
        if feasibility1['feasible'] and not feasibility2['feasible']:
            return True
        
        # 不可行解不支配可行解
        if not feasibility1['feasible'] and feasibility2['feasible']:
            return False
        
        # 两个解都可行或都不可行，比较目标值
        # 对于不可行解，我们可以添加一个轻微的惩罚来鼓励解向可行方向移动
        if not feasibility1['feasible'] and not feasibility2['feasible']:
            # 计算约束违反程度（简单计数）
            violations1 = sum(not v for k, v in feasibility1.items() if k != "feasible")
            violations2 = sum(not v for k, v in feasibility2.items() if k != "feasible")
            
            # 如果第一个解违反的约束更少，则它支配第二个解
            if violations1 < violations2:
                return True
            # 如果第二个解违反的约束更少，则第一个解不支配第二个解
            elif violations1 > violations2:
                return False
        
        # 在可行性相同的情况下，比较目标值
        return (np.all(obj1 <= obj2) and np.any(obj1 < obj2))

    def decision_making_ASF_DR(self, pareto_front: List[Dict]) -> Dict:
        """
        决策制定：从Pareto前沿中选择一个解作为当前决策
        参数:
            pareto_front: Pareto前沿解集
        返回:
            选择的解
        """
        # 采用基于理想点的拐点选择
        best_solution = None
        min_distance = float('inf')

        # 所有解的目标值
        objs = np.array([sol['objectives'] for sol in pareto_front])
        N, M = objs.shape

        # 计算Pareto前沿各个目标的最小值作为理想点
        ideal_point = np.min(objs, axis=0)
        # 计算Pareto前沿各个目标的最大值作为极大点
        nadir_point = np.max(objs, axis=0)

        scale = nadir_point - ideal_point
        # 避免除以0
        scale[scale == 0] = 1e-12

        # 计算归一化的g值
        g = (nadir_point - objs) / scale
        # 计算ASF
        ASF = np.sum(g, axis=1)
        # 计算权重向量w 
        w = g / ASF[:, None]
        
        norms = np.linalg.norm(w, axis=1)
        # 计算余弦相似度矩阵
        cos_sim = (w @ w.T) / (norms[:, None] * norms[None, :])
        # 处理数值稳定性问题
        cos_sim = np.clip(cos_sim, -1.0, 1.0)  # 确保数值稳定性

        A = np.arccos(cos_sim)  # 计算角度矩阵

        neighbors = np.argsort(A, axis=1)[:, 1:int(0.1*N)]  # 排除自身，获取邻居

        U = np.zeros(N)
        for i in range(N):
            # 计算每个解的U值
            U[i] = (ASF[i] * int(0.1 * N)) - ASF[neighbors[i]].sum()
        
        dom_range = np.full(N, np.pi)
        for i in range(N):
            better = np.where(U > U[i])[0]
            if better.size > 0:
                # 计算domination range
                dom_range[i] = A[i, better].min()

        soi_indices = np.argsort(-dom_range)[:1] #  # 选择domination range最大的解作为SOI
        best_solution = pareto_front[soi_indices[0]]

        return best_solution
    
    def decision_making(self, pareto_front: List[Dict]) -> Dict:
        """
        决策制定：从Pareto前沿中选择一个解作为当前决策，选择归一化后距离理想点最近的解
        参数:
            pareto_front: Pareto前沿解集
        返回:
            选择的解
        """
        # 采用基于理想点的拐点选择
        best_solution = None
        min_distance = float('inf')
        # 所有解的目标值
        objs = np.array([sol['objectives'] for sol in pareto_front])
        ideal_point = np.array(self.ideal_point)
        # 计算Pareto前沿各个目标的最小值作为理想点
        ideal_point = np.min(objs, axis=0)
        for solution in pareto_front:
            objectives = np.array(solution['objectives'])
            # 计算归一化距离(注意nadir_point - ideal_point 可能为0，需要避免除以0)
            distance = np.linalg.norm((objectives - ideal_point) / (self.nadir_point - ideal_point + 1e-12))
            if distance < min_distance:
                min_distance = distance
                best_solution = solution

        return best_solution

    def plot_pareto_front(self, pareto_front: List[Dict], title: str = "Pareto Front", objectives_index: List[int] = None, objectives_name: List[str] = None):
        """绘制Pareto前沿
        参数:
            pareto_front: Pareto前沿解集
            title: 图表标题
            objectives_index: 目标函数索引，如果为None则自动选择所有目标（0：车辆数，1：总成本，2：总行驶时间，3：总延迟时间，4：冗余空间均匀度指标）
        """
        import matplotlib.pyplot as plt

        objectives = np.array([sol['objectives'] for sol in pareto_front])

        if objectives_index is None:
            objectives_index = self.objectives_index
        objectives = objectives[:, objectives_index]
        plt.figure(figsize=(10, 6))
        plt.scatter(objectives[:, 0], objectives[:, 1], c='blue', marker='o')
        plt.title(title)
        plt.xlabel(objectives_name[0] if objectives_name else "Objective 1")
        plt.ylabel(objectives_name[1] if objectives_name else "Objective 2")
        plt.grid()
        plt.show()

    def run(self) -> List[Dict]:
        """=== 修改12：在主循环中统计替换次数并打印日志"""
        self.initialize_population(heuristic_ratio=0.2)
        # Output the ratio of feasible solutions
        feasible_count = sum(1 for ind in self.population if ind['feasibility']['feasible'])
        print(f"Feasible solutions in initial population: {feasible_count / self.population_size:.2%}")
        self.generation = 0
        for self.generation in range(self.max_generations):

            for i in range(self.population_size):

                neighbors = self.neighborhood[i]
                p1, p2 = random.sample(neighbors, 2)
                if self.generation >= self.generation_opt and random.random() < 0.5:
                    # 从robust_population中不为None的解中随机选择一个
                    parent1 = self.robust_population[p1]['solution']
                else:
                    parent1 = self.population[p1]['solution']
                if self.generation >= self.generation_opt and random.random() < 0.5:
                    parent2 = self.robust_population[p2]['solution']
                else:
                    parent2 = self.population[p2]['solution']

                # 产生 offspring
                if random.random() < self.crossover_rate:
                    offspring_solution, _ = self._crossover(parent1, parent2)
                else:
                    offspring_solution = copy.deepcopy(parent1)
                    
                if random.random() < self.mutation_rate:
                    offspring_solution = self._mutation(offspring_solution)

                if random.random() <= self.local_search_rate:
                    offspring_solution = self._local_search(offspring_solution, self.weight_vectors[i])

                offspring = self._evaluate_solution(offspring_solution)
                # print(f"Feasibility of individual {i+1}: {offspring['feasibility']['feasible']}")
                total_delay_time, total_capacity_violation = robustness_test_routes(
                    offspring['solution'], self.problem_data, self.travel_time_matrix,
                    self.travel_time_disturbance, self.demand_disturbance, self.test_times
                )
 
                robustness_tested_solution = {
                    'solution': copy.deepcopy(offspring['solution']),
                    'objectives': copy.deepcopy(offspring['objectives']),
                    'total_delay_time': total_delay_time,
                    'total_capacity_violation': total_capacity_violation
                }

                self.update_population(i, offspring)
                self.update_robust_population(i, robustness_tested_solution)
                self.update_external_archive(robustness_tested_solution)

            # Output current progress and average objectives
            if self.robust_population:
                # Extract objectives of all non-dominated solutions
                objectives_array = np.array([sol['objectives'] for sol in self.robust_population if sol is not None])
                avg_objectives = np.mean(objectives_array, axis=0)
                # Calculate robustness metric
                robustness_count = [sol['total_delay_time'] + sol['total_capacity_violation'] for sol in self.robust_population if sol is not None]
                mean_robustness = np.mean(robustness_count) if robustness_count else 0
                print(f"Current generation: {self.generation+1}/{self.max_generations}, Average objectives: {avg_objectives}, Robustness metric: {mean_robustness:.2f}")
            else:
                print(f"Current generation: {self.generation+1}/{self.max_generations}, No non-dominated solutions available")
            

        if len(self.external_robust_archive) == 0:
            # If external robustness archive is empty, return the solution with best robustness metric from current population
            print("External robustness archive is empty, returning the solution with best robustness metric from current population")
            robustness_indicator = [sol['total_delay_time'] + sol['total_capacity_violation'] for sol in self.robust_population if sol is not None]
            if robustness_indicator:
                min_index = np.argmin(robustness_indicator)
                best_robust_solutions = [sol for sol in self.robust_population if sol is not None]
                best_solution = best_robust_solutions[min_index]
                return [best_solution], None
            else:
                print("No robust solutions in current population, returning non-dominated solutions from entire population")
                return self._non_dominated_sort_robust(self.robust_population), None

        return self.external_robust_archive, None

