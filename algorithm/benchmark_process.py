import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def load_data(file_path, customer_num=None):
    """
    读取数据
    :param file_path: 数据文件路径
    :param customer_num: 需要的客户数量（不包括仓库），None表示读取所有客户
    :return: 客户数据（DataFrame）
    """
    data = pd.read_csv(
        file_path,
        header=None,
        # delim_whitespace=True,
        sep='\s+',
        skiprows=8,
        names=['CUST_NO', 'XCOORD', 'YCOORD', 'DEMAND', 'READY_TIME', 'DUE_DATE', 'SERVICE_TIME'],
        dtype={'CUST_NO': int, 'XCOORD': float, 'YCOORD': float,
               'DEMAND': float, 'READY_TIME': float, 'DUE_DATE': float, 'SERVICE_TIME': float}
    )
    if customer_num is not None:
        customer_num = min(customer_num, len(data) - 1)
        indices = [0] + list(range(1, customer_num + 1))
        data = data.iloc[indices].reset_index(drop=True)
    return data

def read_vehicle(file_path):
    """
    读取车辆数量与车辆容量
    :param file_path: 数据文件路径
    :return: (vehicle_num, vehicle_capacity)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    parts = lines[4].strip().split()
    return int(parts[0]), int(parts[-1])

def cal_distance_matrix(data):
    """
    向量化计算两两客户之间的欧氏距离矩阵
    :param data: 客户数据 (DataFrame)
    :return: 距离矩阵 (ndarray, shape=(n, n))
    """
    # coords = data[['XCOORD', 'YCOORD']].to_numpy()
    # # 利用广播：coords[:,None,:] - coords[None,:,:] -> shape (n,n,2)
    # diff = coords[:, None, :] - coords[None, :, :]
    coords_x = data['XCOORD']
    coords_y = data['YCOORD']
    coords = np.stack([coords_x, coords_y], axis=1)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

def calculate_time(data, distance_matrix, customer_list, speed=1.0):
    """
    计算单条路径的服务开始时间列表（含等待）
    :return: ndarray of start-service times
    """
    ready = data['READY_TIME']
    service = data['SERVICE_TIME']

    t = 0.0
    times = [0.0]  # 仓库出发时刻
    for prev, cur in zip(customer_list[:-1], customer_list[1:]):
        travel = distance_matrix[prev, cur] / speed
        t = max(t + travel, ready[cur])
        times.append(t)
        t += service[cur]
    return np.array(times)

def calculate_arrive_time(data, travel_time_matrix, customer_list, speed=1.0):
    """
    计算单条路径的实际到达时间列表（不含等待 + 服务）
    :return: ndarray of arrival times
    """
    ready = data['READY_TIME']
    service = data['SERVICE_TIME']

    t = 0.0
    arrivals = [0.0]
    for prev, cur in zip(customer_list[:-1], customer_list[1:]):
        travel = travel_time_matrix[prev, cur] / speed
        t += travel
        arrivals.append(t)
        # 服务完毕再出发
        t = max(t, ready[cur]) + service[cur]
    return np.array(arrivals)

def calculate_wait_time(data, travel_time_matrix, customer_list, speed=1.0):
    """
    计算单条路径的总等待时间
    :return: 总等待时间
    """
    ready = data['READY_TIME']
    service = data['SERVICE_TIME']

    t = 0.0
    wait_times = [0.0]
    for prev, cur in zip(customer_list[:-1], customer_list[1:]):
        travel = travel_time_matrix[prev, cur] / speed
        t += travel
        wait_time = max(0.0, ready[cur] - t)
        wait_times.append(wait_time)
        # 服务完毕再出发
        t += wait_time + service[cur]
    return np.sum(wait_times)

def calculate_distance(distance_matrix, customer_list):
    """
    计算单条路径的总距离
    """
    return sum(
        distance_matrix[prev, cur]
        for prev, cur in zip(customer_list[:-1], customer_list[1:])
    )

def plot_route(data, customer_list):
    """
    绘制路径方案
    """
    coords = data.loc[customer_list, ['XCOORD', 'YCOORD']].to_numpy()
    plt.figure(figsize=(8, 8))
    plt.scatter(data['XCOORD'], data['YCOORD'], c='blue', label='客户')
    depot = data.loc[0, ['XCOORD', 'YCOORD']]
    plt.scatter(depot['XCOORD'], depot['YCOORD'], c='green', s=100,
                edgecolors='black', label='仓库', zorder=5)
    plt.plot(coords[:, 0], coords[:, 1], c='red', label='路径')
    plt.title('路径图')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    file_path = 'solomon/r101.txt'
    data = load_data(file_path, customer_num=25)
    print(data.head())

    distance_matrix = cal_distance_matrix(data)

    # 示例路径（仓库 -> ... -> 仓库）
    customer_list = [0, 27, 69, 76, 79, 3, 54, 24, 80, 0]
    print("距离矩阵前5×5：\n", distance_matrix[:5, :5])
    print("服务开始时间：", calculate_time(data, distance_matrix, customer_list))
    print("到达时间：    ", calculate_arrive_time(data, distance_matrix, customer_list))
    print("总路程：      ", calculate_distance(distance_matrix, customer_list))

    plot_route(data, customer_list)
