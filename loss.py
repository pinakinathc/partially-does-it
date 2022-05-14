import cv2
import numpy as np
import torch
import torch.nn.functional as F
from qpth.qp import QPFunction


def QP_solver(u, v, return_sim_map=False):
    """
    Input
        x: B x c x m
        y: B x c x n
    """
    m = u.shape[-1]
    n = v.shape[-1]
    nz = m*n
    nbatch = u.shape[0]

    s = (u * F.adaptive_avg_pool1d(v, 1).repeat(1, 1, m)).sum(1) # Shape: B x m
    s = F.relu(s) + 1e-3

    d = (v * F.adaptive_avg_pool1d(u, 1).repeat(1, 1, n)).sum(1) # Shape: B x n
    d = F.relu(d) + 1e-3

    u = u - u.mean(1).unsqueeze(1)
    v = v - v.mean(1).unsqueeze(1)
    u = u.permute(0, 2, 1).unsqueeze(2).repeat(1, 1, n, 1) # shape: B x m x n x c
    v = v.permute(0, 2, 1).unsqueeze(1).repeat(1, m, 1, 1) # shape: B x m x n x c
    cosine_distance_matrix = F.cosine_similarity(u, v, dim=-1) # shape: B x m x n
    cost, flow = emd_inference_qpth(cosine_distance_matrix, s, d, form='L2')

    if return_sim_map:
        return cost, flow
    return cost


def emd_inference_qpth(distance_matrix, weight1, weight2, form='QP', l2_strength=0.0001):
    """
    to use the QP solver QPTH to derive EMD (LP problem),
    one can transform the LP problem to QP,
    or omit the QP term by multiplying it with a small value,i.e. l2_strngth.
    :param distance_matrix: nbatch * element_number * element_number
    :param weight1: nbatch  * weight_number
    :param weight2: nbatch  * weight_number
    :return:
    emd distance: nbatch*1
    flow : nbatch * weight_number *weight_number

    """

    weight1 = (weight1 * weight1.shape[-1]) / weight1.sum(1).unsqueeze(1)
    weight2 = (weight2 * weight2.shape[-1]) / weight2.sum(1).unsqueeze(1)

    nbatch = distance_matrix.shape[0]
    nelement_distmatrix = distance_matrix.shape[1] * distance_matrix.shape[2]

    nelement_weight1 = weight1.shape[1]
    nelement_weight2 = weight2.shape[1]

    Q_1 = distance_matrix.view(-1, 1, nelement_distmatrix).double()

    if form == 'QP':
        # version: QTQ
        Q = torch.bmm(Q_1.transpose(2, 1), Q_1).double().cuda() + 1e-4 * torch.eye(
            nelement_distmatrix).double().cuda().unsqueeze(0).repeat(nbatch, 1, 1)  # 0.00001 *
        p = torch.zeros(nbatch, nelement_distmatrix).double().cuda()
    elif form == 'L2':
        # version: regularizer
        Q = (l2_strength * torch.eye(nelement_distmatrix).double()).cuda().unsqueeze(0).repeat(nbatch, 1, 1)
        p = distance_matrix.view(nbatch, nelement_distmatrix).double()
    else:
        raise ValueError('Unkown form')

    h_1 = torch.zeros(nbatch, nelement_distmatrix).double().cuda()
    h_2 = torch.cat([weight1, weight2], 1).double()
    h = torch.cat((h_1, h_2), 1)

    G_1 = -torch.eye(nelement_distmatrix).double().cuda().unsqueeze(0).repeat(nbatch, 1, 1)
    G_2 = torch.zeros([nbatch, nelement_weight1 + nelement_weight2, nelement_distmatrix]).double().cuda()
    # sum_j(xij) = si
    for i in range(nelement_weight1):
        G_2[:, i, nelement_weight2 * i:nelement_weight2 * (i + 1)] = 1
    # sum_i(xij) = dj
    for j in range(nelement_weight2):
        G_2[:, nelement_weight1 + j, j::nelement_weight2] = 1
    #xij>=0, sum_j(xij) <= si,sum_i(xij) <= dj, sum_ij(x_ij) = min(sum(si), sum(dj))
    G = torch.cat((G_1, G_2), 1)
    A = torch.ones(nbatch, 1, nelement_distmatrix).double().cuda()
    b = torch.min(torch.sum(weight1, 1), torch.sum(weight2, 1)).unsqueeze(1).double()
    flow = QPFunction(verbose=-1)(Q, p, G, h, A, b)

    emd_score = torch.sum((1 - Q_1).squeeze() * flow, 1)
    return emd_score, flow.view(-1, nelement_weight1, nelement_weight2)


def emd_inference_opencv_test(distance_matrix,weight1,weight2):
    distance_list = []
    flow_list = []

    for i in range (distance_matrix.shape[0]):
        cost,flow=emd_inference_opencv(distance_matrix[i],weight1[i],weight2[i])
        distance_list.append(cost)
        flow_list.append(torch.from_numpy(flow))

    emd_distance = torch.Tensor(distance_list).cuda().double()
    flow = torch.stack(flow_list, dim=0).cuda().double()

    return emd_distance, flow


def emd_inference_opencv(cost_matrix, weight1, weight2):
    # cost matrix is a tensor of shape [N,N]
    cost_matrix = cost_matrix.detach().cpu().numpy()

    weight1 = F.relu(weight1) + 1e-5
    weight2 = F.relu(weight2) + 1e-5

    weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
    weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

    cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
    return cost, flow


def cv2_EMD(u, v):
    m = u.shape[-1]
    n = v.shape[-1]
    nz = m*n
    nbatch = u.shape[0]

    s = (u * F.adaptive_avg_pool1d(v, 1).repeat(1, 1, m)).sum(1) # Shape: B x m
    s = F.relu(s) + 1e-3

    d = (v * F.adaptive_avg_pool1d(u, 1).repeat(1, 1, n)).sum(1) # Shape: B x n
    d = F.relu(d) + 1e-3

    u = u - u.mean(1).unsqueeze(1)
    v = v - v.mean(1).unsqueeze(1)
    u = u.permute(0, 2, 1).unsqueeze(2).repeat(1, 1, n, 1) # shape: B x m x n x c
    v = v.permute(0, 2, 1).unsqueeze(1).repeat(1, m, 1, 1) # shape: B x m x n x c
    cosine_distance_matrix = F.cosine_similarity(u, v, dim=-1) # shape: B x m x n

    emd_distance_cv, cv_flow = emd_inference_opencv_test(cosine_distance_matrix, s, d)
    emd_score_cv=((1-cosine_distance_matrix)*cv_flow).sum(-1).sum(-1)
    return emd_score_cv, cv_flow


if __name__ == '__main__':
    u = torch.randn(10, 512, 25).cuda()
    v = torch.randn(10, 512, 25).cuda()

    cv2_score, cv2_flow, cv2_cosine = cv2_EMD(u, v)
    qpth_score, qpth_flow, qpth_cosine = QP_solver(u, v, return_sim_map=True)

    cv_val = cv2_flow
    qpth_val = qpth_flow

    print('emd difference:', (cv_val - qpth_val).abs().max())
    import numpy as np

    print ('cosine difference: ', (cv2_cosine - qpth_cosine).abs().max())
    np.testing.assert_allclose(cv2_cosine.cpu().detach().numpy(), qpth_cosine.cpu().detach().numpy(), rtol=1e-3, atol=1e-4)

    np.testing.assert_allclose(cv_val.cpu().detach().numpy(), qpth_val.cpu().detach().numpy(), rtol=1e-3, atol=1e-4)
