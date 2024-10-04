import numpy as np
import matplotlib.pyplot as plt

# Hàm số và đạo hàm
def f(x):
    return x**2 + 6*x + 8

def grad(x):
    return 2*x + 6

def learning_rate():
    return 0.09
def gradient_descent_lr_scheduler(decay_strategy,initial_learning_rate,decay_rate,decay_steps,iteration):
    if decay_strategy == "Step Decay":
        if iteration%decay_steps == 0:
            return initial_learning_rate - decay_rate
        else:
            return initial_learning_rate
        
    elif decay_strategy == "Exponential Decay":
        return initial_learning_rate*np.exp(-decay_rate*iteration)
    else:
        return initial_learning_rate

# Hàm tính Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Biến khởi tạo cho gradient descent
x = 10
x_values = [x]
y_true = np.array([f(x)])  # Giá trị thật (y_true) chỉ là giá trị của f(x) ban đầu

# Danh sách lưu trữ các giá trị MSE qua từng lần lặp
mse_values = []

# Gradient descent loop
for i in range(1, 100):
    x_new = x - gradient_descent_lr_scheduler("Step Decay",0.1,0.0001,4,i) * grad(x)  # Tính giá trị x mới
    print(gradient_descent_lr_scheduler("Step Decay",0.1,0.0001,4,i))
    x_values.append(x_new)
    
    # Tính MSE với giá trị y thực tế và y dự đoán tại x mới
    y_pred = np.array([f(x_new)])
    mse = mean_squared_error(y_true, y_pred)
    mse_values.append(mse)  # Lưu MSE
    
    print(f"Kết quả lần lặp thứ {i} là {x_new}, MSE: {mse}")
    
    x = x_new  # Cập nhật x

# Vùng x cho đồ thị f(x)
x_range = np.linspace(-15, 5, 400)
y_range = f(x_range)

# Tạo figure với 2 đồ thị
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Biểu đồ 1: Đồ thị hàm f(x) và các bước cập nhật của Gradient Descent
axs[0].plot(x_range, y_range, label='f(x) = x² + 6x + 8', color='blue')
axs[0].scatter(x_values, f(np.array(x_values)), color='red', label='Gradient Descent Steps', zorder=5)
min_x = -3
axs[0].scatter(min_x, f(min_x), color='green', label='Minimum Point', zorder=5)

# Thêm tiêu đề và chú thích
axs[0].set_title("Gradient Descent on f(x)")
axs[0].set_xlabel("x")
axs[0].set_ylabel("f(x)")
axs[0].legend()
axs[0].grid(True)

# Biểu đồ 2: Biểu đồ MSE qua các vòng lặp
axs[1].plot(range(1, 100), mse_values, label='MSE per iteration', color='orange')
axs[1].set_title("Mean Squared Error per iteration")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("MSE")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
