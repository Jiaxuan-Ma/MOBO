import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_column_min(matrix):
    num_cols = len(matrix[0])
    result = []

    for col in range(num_cols):
        min_val = matrix[0][col]
        for row in range(1, len(matrix)):
            if matrix[row][col] < min_val:
                min_val = matrix[row][col]
        result.append(min_val)

    return result

def get_column_max(matrix):
    num_cols = len(matrix[0])
    result = []

    for col in range(num_cols):
        max_val = matrix[0][col]
        for row in range(1, len(matrix)):
            if matrix[row][col] > max_val:
                max_val = matrix[row][col]
        result.append(max_val)

    return result

def calculate_mean(list1, list2):
    result = [(x + y) / 2 for x, y in zip(list1, list2)]
    return result


def plot(pareto_front, targets, target_names):
    fig, ax = plt.subplots()
    ax.plot(pareto_front[target_names[0]], pareto_front[target_names[1]], 'k--')
    ax.scatter(targets[target_names[0]], targets[target_names[1]])
    ax.set_xlabel(target_names[0])
    ax.set_ylabel(target_names[1])
    ax.set_title('Pareto front of visual space')
    plt.show()