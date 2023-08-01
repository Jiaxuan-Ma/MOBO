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

def non_dominated_sorting(fitness_values):
    num_objectives = fitness_values.shape[1]
    num_solutions = fitness_values.shape[0]

    domination_counts = np.zeros(num_solutions, dtype=int)
    dominated_solutions = [[] for _ in range(num_solutions)]
    frontiers = [[]]

    for i in range(num_solutions):
        for j in range(i + 1, num_solutions):
            if np.all(fitness_values[i] <= fitness_values[j]):
                if np.any(fitness_values[i] < fitness_values[j]):
                    domination_counts[j] += 1
                else:
                    dominated_solutions[i].append(j)
            elif np.all(fitness_values[i] >= fitness_values[j]):
                if np.any(fitness_values[i] > fitness_values[j]):
                    domination_counts[i] += 1
                else:
                    dominated_solutions[j].append(i)

        if domination_counts[i] == 0:
            frontiers[0].append(i)

    i = 0
    while len(frontiers[i]) > 0:
        next_frontier = []
        for j in frontiers[i]:
            for k in dominated_solutions[j]:
                domination_counts[k] -= 1
                if domination_counts[k] == 0:
                    next_frontier.append(k)
        i += 1
        frontiers.append(next_frontier)

    return frontiers[:-1]

def find_non_dominated_solutions(fitness_values, **kargs):
    fronts = non_dominated_sorting(fitness_values)
    non_dominated_solutions_idx = []
    for front in fronts:
        non_dominated_solutions_idx.extend(front)
    non_dominated_solutions_Data = fitness_values[non_dominated_solutions_idx]
    non_dominated_solutions_Data = pd.DataFrame(non_dominated_solutions_Data, columns=feature_name)
    non_dominated_solutions_Data.sort_values(by=feature_name[0], inplace=True)

    return  non_dominated_solutions_Data

def dominated_hypervolume(pareto_data, ref_point):
    pareto_data = np.vstack([pareto_data, ref_point])
    pareto_data = pareto_data[np.argsort(-pareto_data[:,0])]
    S = 0
    for i in range(pareto_data.shape[0]-1):
        S += (pareto_data[i,0] - pareto_data[i+1,0]) * (pareto_data[0,1] - pareto_data[i+1,1])
    return S

def plot(pareto_front, targets, target_names):
    fig, ax = plt.subplots()
    ax.plot(pareto_front[target_names[0]], pareto_front[target_names[1]], 'k--')
    ax.scatter(targets[target_names[0]], targets[target_names[1]])
    ax.set_xlabel(target_names[0])
    ax.set_ylabel(target_names[1])
    ax.set_title('Pareto front of visual space')
    plt.show()