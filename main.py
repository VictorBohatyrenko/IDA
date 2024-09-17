import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Функція для генерації точок та кластерів
def generate_clusters(n_points, n_clusters, cluster_std):
    points, labels = make_blobs(n_samples=n_points, centers=n_clusters, cluster_std=cluster_std, random_state=42)
    return points, labels

# Функція для генерації нечіткої матриці розбиття U* з плавними ступенями належності
def generate_fuzzy_partition_matrix(n_clusters, n_points, labels, fuzziness_level=0.5):
    U = np.zeros((n_clusters, n_points))
    for j in range(n_points):
        for k in range(n_clusters):
            # Призначаємо ступені належності з нечіткістю
            U[k, j] = np.random.uniform(0, 1)
        U[:, j] /= np.sum(U[:, j])  # Нормалізуємо, щоб сума ступенів належності дорівнювала 1
    return U

# Функція для додавання шуму до кластеризації (перемішування точок)
def add_noise_to_partition_matrix(U, noise_level):
    noisy_U = U.copy()
    n_clusters, n_points = U.shape
    for j in range(n_points):
        if np.random.rand() < noise_level:
            # Випадково змінюємо ступінь належності
            noisy_U[:, j] = np.random.uniform(0, 1, n_clusters)
            noisy_U[:, j] /= np.sum(noisy_U[:, j])  # Нормалізуємо, щоб сума дорівнювала 1
    return noisy_U

# Функція для розрахунку індексу чіткості CI
def calculate_clarity_index(U):
    n_clusters, n_points = U.shape
    PC = np.sum(U**2) / n_points
    CI = (n_clusters * PC - 1) / (n_clusters - 1)
    return CI

# Функція для створення розбиття при іншій кількості кластерів g
def create_partition_with_different_g(points, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(points)
    U = generate_fuzzy_partition_matrix(n_clusters, len(points), labels)
    return U, labels

# Генеруємо точки з початково заданою кількістю кластерів g*
n_points = 100  # кількість точок
g_star = 3  # початкова кількість кластерів
cluster_std = 0.6  # стандартне відхилення кластерів

points, labels = generate_clusters(n_points, g_star, cluster_std)

# Генеруємо нечітку матрицю розбиття
U_star = generate_fuzzy_partition_matrix(g_star, n_points, labels)

# Обчислюємо індекс чіткості для еталонного розбиття
CI_star = calculate_clarity_index(U_star)

# Створюємо зашумлену кластеризацію
noise_level = 0.2  # рівень шуму
U_noisy = add_noise_to_partition_matrix(U_star, noise_level)

# Обчислюємо індекс чіткості для зашумленого розбиття
CI_noisy = calculate_clarity_index(U_noisy)

# Розглядаємо випадок, коли кількість кластерів g != g*
g_new = 5  # нова кількість кластерів (більше за еталонну)
U_new, new_labels = create_partition_with_different_g(points, g_new)

# Обчислюємо індекс чіткості для розбиття з іншою кількістю кластерів
CI_new = calculate_clarity_index(U_new)

# Виводимо результати індексу чіткості
print("Індекс чіткості для еталонного розбиття (g* = {}): ".format(g_star), CI_star)
print("Індекс чіткості для зашумленого розбиття: ", CI_noisy)
print("Індекс чіткості для кластеризації з g = {}: ".format(g_new), CI_new)

# Побудова трьох графіків для візуалізації різних кластеризацій
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Графік для еталонної кластеризації
axs[0].scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis')
axs[0].set_title("Еталонна кластеризація (g* = {})".format(g_star))

# Графік для зашумленої кластеризації
noisy_labels = np.argmax(U_noisy, axis=0)
axs[1].scatter(points[:, 0], points[:, 1], c=noisy_labels, cmap='viridis')
axs[1].set_title("Зашумлена кластеризація")

# Графік для кластеризації з іншою кількістю кластерів
axs[2].scatter(points[:, 0], points[:, 1], c=new_labels, cmap='viridis')
axs[2].set_title("Кластеризація з g = {}".format(g_new))

# Показуємо графіки
plt.show()
