
from FLDN_utils import *
from sklearn.cluster import KMeans
import time
import math
import numpy as np


graph_path = './data/network.dat'
result_path = './data/observation_data.txt'


mean=0.3
scale=0.05

read_flag=False
update_x_sum_threshold=5
update_p_sum_threshold=5
max_delta_x_threshold=0.005
max_delta_p_threshold=0.005
learning_rate_p=0.001
learning_rate_x=0.005
degree=4
sample_times=15
do_prune=True
epsilon_change=False
ourter_delta_x_threshold=0.001
comb_k=3
x_max_iteration=15
p_max_iteration=15
outer_max_iteration=6
client_num = 5


def local_calculate_gradient_p(p_matrix, s_matrix, prior_network, x_matrix):

    inner_p_cnt = 0
    small_value = 1e-20

    inner_p_cnt += 1

    beta, nodes_num = s_matrix.shape
    p_gradient_matrix = np.zeros(p_matrix.shape)

    for i in range(beta):
        for j in range(nodes_num):

            x_j = x_matrix[:, j].copy()
            gradient_j_zero = np.zeros(nodes_num)
            temp = 1 - p_matrix[:, j] * s_matrix[i]
            temp[np.where(temp == 0)] = np.inf
            gradient_j_zero = -1 * s_matrix[i] / temp * prior_network[:, j] * (1 - s_matrix[i, j]) * x_j
            p_gradient_matrix[:, j] += gradient_j_zero


            gradient_j_one = np.zeros(nodes_num)
            p = p_matrix[:, j].copy()
            s = s_matrix[i].copy()
            A = 1 - np.prod((1 - p * s + small_value) ** x_j)

            down = A * (1 - s * p)
            down[np.where(down == 0)] = np.inf
            up = (-A + 1) * x_j * s * s_matrix[i, j] * prior_network[:, j]
            gradient_j_one = up / down
            p_gradient_matrix[:, j] += gradient_j_one

    return p_gradient_matrix




def local_calculate_gradient_x(p_matrix, s_matrix, prior_network, x_matrix):

    inner_x_cnt = 0
    small_value = 1e-20

    inner_x_cnt += 1

    beta, nodes_num = s_matrix.shape
    x_gradient_matrix = np.zeros(x_matrix.shape)

    for i in range(beta):
        for j in range(nodes_num):



            x_j = x_matrix[:, j].copy()
            gradient_j_zero = np.zeros(nodes_num)
            temp = 1 - p_matrix[:, j] * s_matrix[i]
            gradient_j_zero = prior_network[:, j] * (1 - s_matrix[i, j]) * np.log(temp + small_value)
            x_gradient_matrix[:, j] += gradient_j_zero


            gradient_j_one = np.zeros(nodes_num)
            p = p_matrix[:, j].copy()
            s = s_matrix[i].copy()
            A = 1 - np.prod((1 - p * s + small_value) ** x_j)
            orig_A = A.copy()
            if A == 0:
                A = np.inf

            temp_gradient = prior_network[:, j] * (orig_A - 1) * np.log(
                1 - p_matrix[:, j] * s_matrix[i] + small_value) / A * s_matrix[i, j]
            gradient_j_one = temp_gradient.copy()
            x_gradient_matrix[:, j] += gradient_j_one

    return x_gradient_matrix



def with_x_likelihood_update_x(p_matrix, s_matrix, prior_network, x_matrix, initial_epsilon, iter_cnt, epsilon_change,
                               update_x_sum_threshold, groundtruth_network, sample_times, max_delta_x_threshold):
    inner_x_cnt = 0
    small_value = 1e-20
    pre_x_matrix = x_matrix.copy()
    while True:
        inner_x_cnt += 1

        beta, nodes_num = s_matrix.shape
        x_gradient_matrix = np.zeros(x_matrix.shape)

        for i in range(beta):
            for j in range(nodes_num):

                x_j = x_matrix[:, j].copy()
                gradient_j_zero = np.zeros(nodes_num)
                temp = 1 - p_matrix[:, j] * s_matrix[i]
                gradient_j_zero = prior_network[:, j] * (1 - s_matrix[i, j]) * np.log(temp + small_value)
                x_gradient_matrix[:, j] += gradient_j_zero


                gradient_j_one = np.zeros(nodes_num)
                p = p_matrix[:, j].copy()
                s = s_matrix[i].copy()
                A = 1 - np.prod((1 - p * s + small_value) ** x_j)
                orig_A = A.copy()
                if A == 0:
                    A = np.inf

                temp_gradient = prior_network[:, j] * (orig_A - 1) * np.log(
                    1 - p_matrix[:, j] * s_matrix[i] + small_value) / A * s_matrix[i, j]
                gradient_j_one = temp_gradient.copy()
                x_gradient_matrix[:, j] += gradient_j_one


        if epsilon_change:
            epsilon = initial_epsilon / np.sqrt(iter_cnt)
        else:
            epsilon = initial_epsilon

        x_matrix += epsilon * x_gradient_matrix


        x_matrix[np.where(x_matrix < 0)] = 0
        x_matrix[np.where(x_matrix > 1)] = 1


        show_update_x(groundtruth_network, x_matrix, p_matrix, s_matrix, sample_times, inner_x_cnt)
        max_delta_x = np.max(abs(pre_x_matrix - x_matrix))
        delta_x_sum = np.sum(abs(pre_x_matrix - x_matrix))
        if delta_x_sum < update_x_sum_threshold or max_delta_x < max_delta_x_threshold or inner_x_cnt > 30:
            break

        pre_x_matrix = x_matrix.copy()

    return x_matrix


def local_mi(diffusion_results):

    results_num, nodes_num = diffusion_results.shape

    local_MI = np.zeros((4,nodes_num, nodes_num))

    for j in range(1,nodes_num):
        for k in range(j):
            for result_index in range(results_num):
                local_MI[0,j,k] += (1 - diffusion_results[result_index, j]) * (1 - diffusion_results[result_index, k])
                local_MI[1,j,k] += (1 - diffusion_results[result_index, j]) * diffusion_results[result_index, k]
                local_MI[2,j,k] += diffusion_results[result_index, j] * (1 - diffusion_results[result_index, k])
                local_MI[3,j,k] += diffusion_results[result_index, j] * diffusion_results[result_index, k]

    return local_MI,results_num


def aggregate_mi(mi_list,data_num_list):

    MI = np.zeros(mi_list[0][0].shape)
    nodes_num = MI.shape[0]
    clients_num = len(data_num_list)
    results_num = sum(data_num_list)

    for i in range(1,clients_num):
        mi_list[0] = mi_list[0] + mi_list[i]

    for j in range(1,nodes_num):
        for k in range(j):
            epsilon = 1e-5
            M00 = mi_list[0][0,j,k] / results_num * math.log(
                mi_list[0][0,j,k] * results_num / (mi_list[0][0,j,k] + mi_list[0][1,j,k]) / (
                        mi_list[0][0,j,k] + mi_list[0][2,j,k]) + epsilon, 2)
            M01 = mi_list[0][1,j,k] / results_num * math.log(
                mi_list[0][1,j,k] * results_num / (mi_list[0][0,j,k] + mi_list[0][1,j,k]) / (
                        mi_list[0][1,j,k] + mi_list[0][3,j,k]) + epsilon, 2)
            M10 = mi_list[0][2,j,k] / results_num * math.log(
                mi_list[0][2,j,k] * results_num / (mi_list[0][2,j,k] + mi_list[0][3,j,k]) / (
                        mi_list[0][0,j,k] + mi_list[0][2,j,k]) + epsilon, 2)
            M11 = mi_list[0][3,j,k] / results_num * math.log(
                mi_list[0][3,j,k] * results_num / (mi_list[0][2,j,k] + mi_list[0][3,j,k]) / (
                        mi_list[0][1,j,k] + mi_list[0][3,j,k]) + epsilon, 2)


            MI[j, k] = M00 + M11 - abs(M10) - abs(M01)


            MI[k, j] = MI[j, k]


    MI[np.where(MI < 0)] = 0
    tmp_MI = MI.reshape((-1, 1))

    estimator = KMeans(n_clusters=2)
    estimator.fit(tmp_MI)
    label_pred = estimator.labels_
    temp_0 = tmp_MI[label_pred == 0]
    temp_1 = tmp_MI[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)

    prune_network = np.zeros((nodes_num, nodes_num))
    prune_network[np.where(MI > tau)] = 1

    return prune_network




def aggregate_para(para_matrix):

    avg_matrix = np.zeros(para_matrix[0].shape)
    client_num = len(para_matrix)
    for i in range(client_num):
        avg_matrix = avg_matrix + para_matrix[i]
    avg_matrix = avg_matrix/client_num
    return avg_matrix




def update_p_with_globgrad(p_matrix, p_gradient_matrix, initial_epsilon, iter_cnt,
                               epsilon_change,
                               update_p_sum_threshold, groundtruth_network, groundtruth_p, max_delta_p_threshold,
                               p_max_iteration):

    inner_p_cnt = 0
    small_value = 1e-20
    pre_p_matrix = p_matrix.copy()
    while True:
        begin_1 = time.time()
        inner_p_cnt += 1


        if epsilon_change:
            epsilon = initial_epsilon / np.sqrt(iter_cnt)
        else:
            epsilon = initial_epsilon

        p_matrix += epsilon * p_gradient_matrix


        p_matrix[np.where(p_matrix < 0)] = 0
        p_matrix[np.where(p_matrix > 1)] = 1

        end_1 = time.time()
        print("this iteration time cost=%f" % (end_1 - begin_1))


        show_update_p(groundtruth_network, groundtruth_p, p_matrix, inner_p_cnt)
        max_delta_p = np.max(abs(pre_p_matrix - p_matrix))
        delta_p_sum = np.sum(abs(pre_p_matrix - p_matrix))
        if delta_p_sum < update_p_sum_threshold or max_delta_p < max_delta_p_threshold or inner_p_cnt >= p_max_iteration:
            break

        pre_p_matrix = p_matrix.copy()

    return p_matrix


def update_x_with_globgrad( x_matrix, x_gradient_matrix, initial_epsilon, iter_cnt,
                                       epsilon_change, update_x_sum_threshold, groundtruth_network, sample_times,
                                       comb_k, max_delta_x_threshold, x_max_iteration):

    inner_x_cnt = 0
    small_value = 1e-20
    pre_x_matrix = x_matrix.copy()
    while True:
        begin_1 = time.time()
        inner_x_cnt += 1

        if epsilon_change:
            epsilon = initial_epsilon / np.sqrt(iter_cnt)
        else:
            epsilon = initial_epsilon

        x_matrix += epsilon * x_gradient_matrix


        x_matrix[np.where(x_matrix < 0)] = 0
        x_matrix[np.where(x_matrix > 1)] = 1

        end_1 = time.time()
        print("this iteration time cost=%f" % (end_1 - begin_1))


        max_delta_x = np.max(abs(pre_x_matrix - x_matrix))
        delta_x_sum = np.sum(abs(pre_x_matrix - x_matrix))
        if delta_x_sum < update_x_sum_threshold or max_delta_x < max_delta_x_threshold or inner_x_cnt >= x_max_iteration:
            break

        pre_x_matrix = x_matrix.copy()

    return x_matrix


def adaptive_threshold(MI):
    nodes_num = MI.shape[0]
    MI[np.where(MI < 0)] = 0
    tmp_MI = MI.reshape((-1, 1))

    estimator = KMeans(n_clusters=2)
    estimator.fit(tmp_MI)
    label_pred = estimator.labels_
    temp_0 = tmp_MI[label_pred == 0]
    temp_1 = tmp_MI[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        rho = np.max(temp_1)
    else:
        rho = np.max(temp_0)

    prune_network = np.zeros((nodes_num, nodes_num))
    print("rho = %f" % (rho))
    prune_network[np.where(MI > rho)] = 1

    for i in range(nodes_num):
        prune_network[i, i] = 0

    return prune_network


def FLDN():

    begin = time.time()
    overall_begin = time.time()

    ground_truth_network, diffusion_result = load_data(graph_path, result_path)
    nodes_num = diffusion_result.shape[1]
    ground_truth_p = ground_truth_network * 0.3

    split_result = np.split(ground_truth_network,client_num,axis=0)



    local_mi_list = []
    data_num_list = []
    for i in range(client_num):
        local_mi_result, data_num = local_mi(split_result[i])
        local_mi_list.append(local_mi_result)
        data_num_list.append(data_num)

    Fed_prune_network = aggregate_mi(local_mi_list, data_num_list)


    x_coe = 1e-5
    p_coe = 1e-5

    x_matrix = np.random.rand(nodes_num, nodes_num) * x_coe
    x_matrix[np.where(Fed_prune_network == 1)] = 1

    p_matrix = np.random.rand(nodes_num, nodes_num)
    p_matrix[np.where(Fed_prune_network == 0)] *= p_coe

    prior_network = np.ones((nodes_num, nodes_num))


    it_cnt = 0
    pre_x = x_matrix.copy()
    while True:
        it_cnt += 1

        outer_begin = time.time()

        para_list = []
        for c in range(client_num):

            p_gradient_matrix = local_calculate_gradient_p(p_matrix, split_result[c], prior_network, x_matrix)
            para_list.append(p_gradient_matrix)

        p_globgrad_matrix = aggregate_para(para_list)
        p_matrix = update_p_with_globgrad(p_matrix,p_globgrad_matrix, learning_rate_p,
                                                  it_cnt,
                                                  epsilon_change, update_p_sum_threshold, ground_truth_network,
                                                  ground_truth_p, max_delta_p_threshold, p_max_iteration)

        p_end = time.time()
        show_update_p(ground_truth_network, ground_truth_p, p_matrix, it_cnt)

        para_list = []
        for c in range(client_num):
            x_gradient_matrix = local_calculate_gradient_x(x_matrix, split_result[c], prior_network, p_matrix)
            para_list.append(x_gradient_matrix)

        x_globgrad_matrix = aggregate_para(para_list)
        x_matrix = update_x_with_globgrad( prior_network, x_matrix,x_globgrad_matrix,
                                                          learning_rate_x,
                                                          it_cnt, epsilon_change, update_x_sum_threshold,
                                                          ground_truth_network, sample_times, comb_k,
                                                          max_delta_x_threshold,
                                                          x_max_iteration)
        x_end = time.time()
        show_update_x(ground_truth_network,x_matrix,p_matrix,diffusion_result,sample_times,it_cnt)
        show_update_p(ground_truth_network, ground_truth_p,p_matrix,it_cnt)
        x_integer_matrix = adaptive_threshold(x_matrix)
        print(cal_F1(ground_truth_network, x_integer_matrix))


        max_delta_x = np.max(abs(pre_x - x_matrix))
        if max_delta_x < ourter_delta_x_threshold or it_cnt >= outer_max_iteration:
            print("algorithm done!")
            break

        pre_x = x_matrix.copy()

    x_integer_matrix = adaptive_threshold(x_matrix)
    show_update_p(ground_truth_network, ground_truth_p, p_matrix, it_cnt)
    print(cal_F1(ground_truth_network, x_integer_matrix))







if __name__ == '__main__':
    FLDN()



