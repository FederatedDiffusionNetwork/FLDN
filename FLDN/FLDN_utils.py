
import numpy as np



def cal_loss(x_edge_matrix, p_matrix, prob_result):
    first_item=1-prob_result
    second_item=np.zeros(prob_result.shape)

    epsilon=1e-10
    beta,nodes_num=prob_result.shape
    for record_index in range(beta):
        temp_third = x_edge_matrix.copy()
        prob_l=prob_result[record_index,:].copy()
        for i in range(nodes_num):
            temp_third[i, i] = 0

        temp_third = temp_third * np.log(1 - prob_l.reshape((-1, 1)) * p_matrix + epsilon)
        sum_item = np.sum(temp_third, axis=0).reshape((1, -1))
        second_item[record_index,:]=sum_item.copy()

    loss=np.sum(np.square(first_item-second_item))

    return loss


def show_result(x_edge_matrix_list, p_matrix, prob_result, ground_truth_network):
    loss_list=[]
    for i in range(len(x_edge_matrix_list)):
        cur_matrix=x_edge_matrix_list[i].copy()
        cur_loss=cal_loss(cur_matrix,p_matrix,prob_result)
        loss_list.append(cur_loss)
    max_index=np.argmax(np.array(loss_list))
    max_edge=x_edge_matrix_list[max_index]
    precision, recall, f_score=cal_F1(ground_truth_network,max_edge)

    return precision, recall, f_score

def cal_F1(ground_truth_network, inferred_network):
    TP = np.sum(ground_truth_network + inferred_network == 2)
    FP = np.sum(ground_truth_network - inferred_network == -1)
    FN = np.sum(ground_truth_network - inferred_network == 1)
    print("false edges num = %d" %(FP+FN))
    epsilon = 1e-20
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f_score = 2 * precision * recall / (precision + recall + epsilon)

    return precision, recall, f_score

def modify_p(groundtruth_network, infer_p):
    edges_num = np.sum(groundtruth_network)
    temp = infer_p * groundtruth_network
    mean_p = np.sum(temp) / edges_num
    modified_p = infer_p / mean_p * 0.3

    return modified_p


def cal_mae(groundtruth_network, p, infer_p):
    gt_p = groundtruth_network * p
    edges_num = np.sum(groundtruth_network)
    temp = gt_p.copy()
    temp[temp == 0] = 1
    temp_infer_p = infer_p.copy()
    temp_infer_p = groundtruth_network * temp_infer_p

    mae = np.sum(abs(temp_infer_p - gt_p) / temp) / edges_num

    return mae


def cal_mae_v2(groundtruth_network, p, infer_p):
    gt_p = groundtruth_network * p
    edges_num = np.sum(groundtruth_network)
    temp_infer_p = groundtruth_network * infer_p

    mae_v2 = np.sum(abs(temp_infer_p - gt_p)) / edges_num

    return mae_v2

def cal_mse(p, infer_p):
    mse = np.mean(np.square(p-infer_p))
    return mse


def cal_mse_v2(groundtruth_network, p, infer_p):
    gt_p = groundtruth_network*p
    edges_num = np.sum(groundtruth_network)
    mse_v2=np.sum(np.square(groundtruth_network*infer_p-gt_p))/edges_num

    return mse_v2



def show_update_p(ground_truth_network, ground_truth_p, p_matrix, iter_cnt):
    print("inner_p_cnt:%d" % (iter_cnt))
    mae = cal_mae(ground_truth_network, ground_truth_p, p_matrix)
    mse = cal_mse(ground_truth_p, p_matrix)
    mae_v2 = cal_mae_v2(ground_truth_network, ground_truth_p, p_matrix)
    mse_v2 = cal_mse_v2(ground_truth_network, ground_truth_p, p_matrix)
    print("MAE=%f, MSE=%f, MAE_v2=%f, MSE_v2=%f" % (mae, mse, mae_v2, mse_v2))

    modified_p = modify_p(ground_truth_network, p_matrix)
    modified_mae = cal_mae(ground_truth_network, ground_truth_p, modified_p)
    modified_mse = cal_mse(ground_truth_p, modified_p)
    modified_mae_v2 = cal_mae_v2(ground_truth_network, ground_truth_p, modified_p)
    modified_mse_v2 = cal_mse_v2(ground_truth_network, ground_truth_p, modified_p)
    print("modified_MAE=%f, modified_MSE=%f, modified_MAE_v2=%f, modified_MSE_v2=%f" % (
    modified_mae, modified_mse, modified_mae_v2, modified_mse_v2))

    mae_all = cal_mae_all(ground_truth_p, p_matrix)
    print("mae_all=%f" % (mae_all))


def cal_mae_all(p, infer_p):
    return np.mean(abs(p - infer_p))


def show_update_x(ground_truth_network, x_matrix, p_matrix, prob_result, sample_times, iter_cnt):
    print("inner_x_cnt:%d" % (iter_cnt))

    mae = cal_mae(ground_truth_network, ground_truth_network, x_matrix)
    mse = cal_mse(ground_truth_network, x_matrix)
    mae_v2 = cal_mae_v2(ground_truth_network, ground_truth_network, x_matrix)
    mse_v2 = cal_mse_v2(ground_truth_network, ground_truth_network, x_matrix)
    print("x_MAE=%f, x_MSE=%f, x_MAE_v2=%f, x_MSE_v2=%f" % (mae, mse, mae_v2, mse_v2))
    mae_all = cal_mae_all(ground_truth_network, x_matrix)
    print("x_mae_all=%f" % (mae_all))

    x_matrix_list = sample_x(x_matrix, sample_times)
    precision, recall, f1 = show_result(x_matrix_list, p_matrix, prob_result, ground_truth_network)
    print("precision=%f,recall=%f,f1=%f" % (precision, recall, f1))

def sample_x(x_edge_matrix, sample_times):

    x_matrix_list = []
    for i in range(sample_times):
        cur_x=np.zeros(x_edge_matrix.shape)
        sample = np.random.rand(*x_edge_matrix.shape)
        one_index = np.where(sample < x_edge_matrix)
        cur_x[one_index] = 1
        x_matrix_list.append(cur_x.copy())

    return x_matrix_list


def load_data(graph_path, result_path):
    with open(result_path, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        diffusion_result = np.array([[int(state) for state in line] for line in lines])

    nodes_num = diffusion_result.shape[1]

    with open(graph_path, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        data = np.array([[int(node) for node in line] for line in lines])
        ground_truth_network = np.zeros((nodes_num, nodes_num))
        edges_num = data.shape[0]
        for i in range(edges_num):
            ground_truth_network[data[i, 0] - 1, data[i, 1] - 1] = 1

    return ground_truth_network, diffusion_result
