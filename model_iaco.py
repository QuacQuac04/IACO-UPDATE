import random
import numpy as np
import matplotlib.pyplot as plt

class Customer:
    def __init__(self, id, x, y, demand, ready_time, due_date, service_time):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time

class Vehicle:
    def __init__(self, capacity):
        self.capacity = capacity
        self.route = []
        self.current_load = 0
        self.current_time = 0
        self.is_used = False

    def __str__(self):
        if not self.is_used:
            return "Vehicle not used"
        return "Route: " + " -> ".join([str(customer.id) if not isinstance(customer, Depot) else "Depot" for customer in self.route])

    def add_customer(self, customer):
        """Thêm khách hàng vào tuyến đường"""
        self.route.append(customer)
        self.current_load += customer.demand
        self.current_time += customer.service_time

class VRPTW:
    def __init__(self, customers, depot, max_iterations, num_ants, alpha, beta, rho, num_vehicles):
        self.customers = customers
        self.depot = depot
        self.max_iterations = max_iterations
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = 200  # Từ dữ liệu Solomon
        
        # Tìm ID lớn nhất trong danh sách khách hàng
        self.max_customer_id = max(customer.id for customer in customers) + 1
        
        # Khởi tạo ma trận pheromone và heuristic với kích thước phù hợp
        self.pheromone = np.ones((self.max_customer_id, self.max_customer_id))
        self.heuristic = self.initialize_heuristic()
        
        self.best_solution = None
        self.best_vehicle_count = float('inf')
        self.best_distance = float('inf')

    def initialize_heuristic(self):
        """Khởi tạo ma trận thông tin heuristic dựa trên khoảng cách và thời gian"""
        heuristic = np.zeros((self.max_customer_id, self.max_customer_id))
        
        # Tạo dictionary để map id với customer object
        customer_dict = {customer.id: customer for customer in self.customers}
        
        for i in range(self.max_customer_id):
            for j in range(self.max_customer_id):
                if i != j and i in customer_dict and j in customer_dict:
                    customer1 = customer_dict[i]
                    customer2 = customer_dict[j]
                    # Kết hợp khoảng cách và cửa sổ thời gian
                    distance = self.distance(customer1, customer2)
                    time_window_penalty = abs(customer1.due_date - customer2.ready_time)
                    heuristic[i][j] = 1.0 / (distance + time_window_penalty)
        return heuristic

    def distance(self, customer1, customer2):
        return np.sqrt((customer1.x - customer2.x) ** 2 + (customer1.y - customer2.y) ** 2)

    def evaluate_solution(self, solution):
        total_distance = 0
        total_vehicles = 0
        for vehicle in solution:
            if vehicle.is_used:  # Chỉ tính cho xe được sử dụng
                total_distance += self.calculate_route_distance(vehicle.route)
                total_vehicles += 1
        return total_distance, total_vehicles

    def calculate_route_distance(self, route):
        distance = 0
        current_time = 0
        for i in range(len(route)):
            if isinstance(route[i], Depot):
                continue
            if i == 0:
                distance += self.distance(self.depot, route[i])
            else:
                distance += self.distance(route[i-1], route[i])
            current_time += route[i].service_time
        distance += self.distance(route[-1], self.depot)
        return distance

    def calculate_total_distance(self, solution):
        """Tính tổng khoảng cách cho một giải pháp"""
        total_distance = 0
        for vehicle in solution:
            if vehicle.is_used:
                total_distance += self.calculate_route_distance(vehicle.route)
        return total_distance

    def select_next_customer(self, unvisited, probabilities):
        """Chọn khách hàng tiếp theo dựa trên xác suất"""
        return np.random.choice(unvisited, p=probabilities)

    def is_feasible(self, vehicle, customer):
        """Kiểm tra tính khả thi khi thêm khách hàng vào xe"""
        # Kiểm tra ràng buộc về tải trọng
        if vehicle.current_load + customer.demand > vehicle.capacity:
            return False
        
        # Kiểm tra ràng buộc về thời gian
        if len(vehicle.route) == 0:
            arrival_time = self.distance(self.depot, customer)
        else:
            last_customer = vehicle.route[-1]
            arrival_time = vehicle.current_time + self.distance(last_customer, customer)
        
        # Kiểm tra cửa sổ thời gian
        if arrival_time > customer.due_date:
            return False
        
        service_start = max(arrival_time, customer.ready_time)
        if service_start + customer.service_time > customer.due_date:
            return False
            
        return True

    def local_search(self, solution):
        """Cải thiện giải pháp bằng tìm kiếm cục bộ"""
        improved = True
        while improved:
            improved = False
            # 2-opt trong mỗi tuyến đường
            for vehicle in solution:
                if not vehicle.is_used:
                    continue
                    
                route = vehicle.route
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        # Đảo ngược đoạn từ i đến j
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        # Tính khoảng cách mới
                        old_distance = self.calculate_route_distance(route)
                        new_distance = self.calculate_route_distance(new_route)
                        
                        if new_distance < old_distance:
                            vehicle.route = new_route
                            improved = True

    def construct_solution(self):
        """Xây dựng giải pháp mới sử dụng chiến lược ACO cải tiến"""
        vehicles = [Vehicle(capacity=self.vehicle_capacity) for _ in range(self.num_vehicles)]
        unvisited = self.customers.copy()
        current_vehicle_idx = 0

        while unvisited and current_vehicle_idx < len(vehicles):
            vehicle = vehicles[current_vehicle_idx]
            current = self.depot
            
            while unvisited:
                # Tính xác suất chọn khách hàng tiếp theo
                probabilities = self.calculate_probabilities(current, unvisited, vehicle)
                if not any(probabilities):  # Không tìm thấy khách hàng phù hợp
                    break
                    
                # Chọn khách hàng tiếp theo dựa trên xác suất
                next_customer = self.select_next_customer(unvisited, probabilities)
                
                # Kiểm tra ràng buộc
                if self.is_feasible(vehicle, next_customer):
                    vehicle.add_customer(next_customer)
                    unvisited.remove(next_customer)
                    current = next_customer
                else:
                    break
            
            if vehicle.route:  # Nếu xe có phục vụ ít nhất 1 khách hàng
                vehicle.is_used = True
                vehicle.route.append(self.depot)
            current_vehicle_idx += 1

        return vehicles

    def calculate_probabilities(self, current, unvisited, vehicle):
        """Tính xác suất chọn khách hàng tiếp theo dựa trên pheromone và heuristic"""
        probabilities = []
        total = 0
        
        for customer in unvisited:
            if not self.is_feasible(vehicle, customer):
                probabilities.append(0)
                continue
                
            # Tính toán hấp dẫn của đường đi
            if isinstance(current, Depot):
                # Nếu đang ở depot, sử dụng giá trị pheromone và heuristic đặc biệt
                pheromone_value = 1.0  # Giá trị mặc định cho depot
                heuristic_value = 1.0 / (self.distance(self.depot, customer) + 1)
            else:
                pheromone_value = self.pheromone[current.id][customer.id]
                heuristic_value = self.heuristic[current.id][customer.id]
            
            # Công thức ACO cải tiến với trọng số alpha và beta
            attraction = (pheromone_value ** self.alpha) * (heuristic_value ** self.beta)
            probabilities.append(attraction)
            total += attraction
            
        # Chuẩn hóa xác suất
        if total > 0:
            probabilities = [p/total for p in probabilities]
        
        return probabilities

    def update_pheromone(self, solutions):
        """Cập nhật pheromone với cơ chế bay hơi và tăng cường"""
        # Bay hơi pheromone
        self.pheromone *= (1 - self.rho)
        
        # Tăng cường pheromone cho các đường đi tốt
        for solution in solutions:
            vehicles_used = sum(1 for v in solution if v.is_used)
            total_distance = self.calculate_total_distance(solution)
            
            # Ưu tiên số lượng xe (f1) trong việc tăng cường pheromone
            vehicle_factor = 1.0 / (vehicles_used + 1)  # +1 để tránh chia cho 0
            distance_factor = 1.0 / (total_distance + 1)
            
            # Tăng cường pheromone với trọng số ưu tiên
            delta_pheromone = 2 * vehicle_factor + distance_factor
            
            for vehicle in solution:
                if not vehicle.is_used:
                    continue
                    
                for i in range(len(vehicle.route) - 1):
                    if isinstance(vehicle.route[i], Depot) or isinstance(vehicle.route[i + 1], Depot):
                        continue
                    self.pheromone[vehicle.route[i].id][vehicle.route[i + 1].id] += delta_pheromone

    def run(self):
        """Chạy thuật toán IACO"""
        for iteration in range(self.max_iterations):
            solutions = []
            
            # Mỗi kiến xây dựng một giải pháp
            for _ in range(self.num_ants):
                solution = self.construct_solution()
                solutions.append(solution)
                
                # Đánh giá giải pháp
                vehicles_used = sum(1 for v in solution if v.is_used)
                total_distance = self.calculate_total_distance(solution)
                
                # Cập nhật giải pháp tốt nhất với ưu tiên f1 (số xe)
                if (vehicles_used < self.best_vehicle_count) or \
                   (vehicles_used == self.best_vehicle_count and total_distance < self.best_distance):
                    self.best_vehicle_count = vehicles_used
                    self.best_distance = total_distance
                    self.best_solution = solution
            
            # Cập nhật pheromone
            self.update_pheromone(solutions)
            
            # Local search để cải thiện giải pháp tốt nhất
            self.local_search(self.best_solution)

class Depot:
    def __init__(self, x, y):
        self.id = 0  # Thêm id cho depot
        self.x = x
        self.y = y

def read_solomon_data(file_path):
    customers = []
    depot = None
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if not line.strip() or line.startswith('C') or line.startswith('VEHICLE') or line.startswith('CUSTOMER'):
                continue
            
            data = line.split()
            if depot is None and len(data) >= 3:
                depot = Depot(float(data[1]), float(data[2]))
            elif len(data) >= 7:
                customers.append(Customer(
                    id=int(data[0]),
                    x=float(data[1]),
                    y=float(data[2]),
                    demand=float(data[3]),
                    ready_time=float(data[4]),
                    due_date=float(data[5]),
                    service_time=float(data[6])
                ))
    return customers, depot

def calculate_distance(customer1, customer2):
    return np.sqrt((customer1.x - customer2.x) ** 2 + (customer1.y - customer2.y) ** 2)

# Đọc dữ liệu từ tệp Solomon
customers, depot = read_solomon_data('path_to_solomon_data3.txt')

# Ví dụ tính toán khoảng cách giữa kho và khách hàng đầu tiên
distance_to_first_customer = calculate_distance(depot, customers[0])
print(f"Distance from depot to customer 0: {distance_to_first_customer:.2f}")

# Cài đặt tham số
max_iterations = 100    # Số vòng lặp tối đa
num_ants = 50          # Số lượng kiến trong quần thể
alpha = 1              # Trọng số của pheromone
beta = 2               # Trọng số của thông tin heuristic  
rho = 0.9             # Tỷ lệ bay hơi pheromone
num1 = 20             # Số lượng thành phố ưu tiên trong khởi tạo

# Khởi chạy thuật toán
vrptw = VRPTW(customers, depot, max_iterations, num_ants, alpha, beta, rho, num1)
vrptw.run()

# In kết quả
print("Chi phi:", vrptw.best_distance)
print("So luong xe tot nhat:", vrptw.best_vehicle_count)
print("Duong di toi uu:")
for vehicle in vrptw.best_solution:
    if vehicle.is_used:  # Chỉ in ra các xe được sử dụng
        print(vehicle)

# Vẽ lộ trình
plt.figure(figsize=(10, 6))
for vehicle in vrptw.best_solution:
    if vehicle.is_used:  # Chỉ vẽ các xe được sử dụng
        route_x = [depot.x] + [customer.x for customer in vehicle.route if not isinstance(customer, Depot)] + [depot.x]
        route_y = [depot.y] + [customer.y for customer in vehicle.route if not isinstance(customer, Depot)] + [depot.y]
        plt.plot(route_x, route_y, marker='o')
        for customer in vehicle.route:
            if not isinstance(customer, Depot):
                plt.text(customer.x, customer.y, str(customer.id), fontsize=12, ha='right')

plt.title('Tuyến Đường Đi Tốt Nhất')
plt.xlabel('Trục X')
plt.ylabel('Trục Y')
plt.grid()
plt.show()