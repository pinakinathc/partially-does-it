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
    similarity_map = F.cosine_similarity(u, v, dim=-1) # shape: B x m x n
    cost, flow = emd_inference_qpth(similarity_map, s, d)

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


def EMD_similarity(x, y, return_sim_map=False):
    """ 
    Input 
        x: B x 512 x 7 x 7
        y: B x 512 x 7 x 7
    Return
        EMD distance calculated using cv2
    """

    weight1 = get_weight_vector(x, y)
    weight2 = get_weight_vector(y, x)

    # print ('shape of weight1: {}, weight2: {}'.format(
    #     weight1.shape, weight2.shape
    # ))

    # normalise
    x = x - x.mean(1).unsqueeze(1)
    y = y - y.mean(1).unsqueeze(1)

    similarity_map = get_similiarity_map(y, x)
    
    for i in range(x.shape[0]):
        _, flow = emd_inference_opencv(1 - similarity_map[i, i, :, :], weight1[i, i, :], weight2[i, i, :])
        print (flow)
        similarity_map[i, i, :, :] = (
            similarity_map[i, i, :] * torch.from_numpy(flow).to(similarity_map.device))

    num_node = weight1.shape[-1]
    distance = similarity_map.sum(-1).sum(-1) * (12.5/num_node) # Shape: B x B

    if return_sim_map:
        return distance.diag(), torch.cat([similarity_map[i, i, :, :] for i in range(x.shape[0])])
    return distance.diag() # B x 1


def get_weight_vector(A, B):

    M = A.shape[0]
    N = B.shape[0]

    B = F.adaptive_avg_pool2d(B, [1, 1])
    B = B.repeat(1, 1, A.shape[2], A.shape[3])

    A = A.unsqueeze(1)
    B = B.unsqueeze(0)

    A = A.repeat(1, N, 1, 1, 1)
    B = B.repeat(M, 1, 1, 1, 1)

    combination = (A * B).sum(2)
    combination = combination.view(M, N, -1)
    combination = F.relu(combination) + 1e-3
    return combination


def get_similiarity_map(proto, query):
    way = proto.shape[0]
    num_query = query.shape[0]
    query = query.view(query.shape[0], query.shape[1], -1)
    proto = proto.view(proto.shape[0], proto.shape[1], -1)

    proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
    query = query.unsqueeze(1).repeat([1, way, 1, 1])
    proto = proto.permute(0, 1, 3, 2)
    query = query.permute(0, 1, 3, 2)
    feature_size = proto.shape[-2]

    proto = proto.unsqueeze(-3)
    query = query.unsqueeze(-2)
    query = query.repeat(1, 1, 1, feature_size, 1)
    similarity_map = F.cosine_similarity(proto, query, dim=-1)

    return similarity_map


def emd_inference_opencv(cost_matrix, weight1, weight2):
    # cost matrix is a tensor of shape [N,N]
    cost_matrix = cost_matrix.detach().cpu().numpy()

    weight1 = F.relu(weight1) + 1e-5
    weight2 = F.relu(weight2) + 1e-5

    weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
    weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

    # print ('Shape of weight1: {}, weight2: {}, cost_matrix: {}'.format(
    # 	weight1.shape, weight2.shape, cost_matrix.shape))
    cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
    return cost, flow


def opencv_Solver(u, v):
# if __name__ == '__main__':
    """
    Input
        x: B x c x m
        y: B x c x n
    """
    # u = torch.randn(7, 512, 9).cuda()
    # v = torch.randn(7, 512, 9).cuda()

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
    similarity_map = F.cosine_similarity(u, v, dim=-1) # shape: B x m x n

    total_flow  = torch.zeros_like(similarity_map, dtype=torch.float32).cuda()

    for b in range(similarity_map.shape[0]):
        _, _, flow = cv2.EMD(
            s[b].detach().cpu().numpy(),
            d[b].detach().cpu().numpy(),
            cv2.DIST_USER,
            similarity_map[b].detach().cpu().numpy()
        )
        total_flow[b] = torch.tensor(flow)

    cost = torch.sum(torch.sum((1 - similarity_map).squeeze() * total_flow, 1), 1)
    return cost
























if __name__ == '__main__':
    x = torch.randn(1, 512, 5, 5).cuda()
    y = torch.randn(1, 512, 3,3).cuda()

    dist = EMD_similarity(x, y)
    print ('Simplex dist: ', dist)

    x = x.reshape(1, 512, -1) # shape: B x c x m
    y = y.reshape(1, 512, -1) # shape: B x c x n

    cost = QP_solver(x, y)
    input('cost: {}'.format(cost))

    m = x.shape[-1]
    n = y.shape[-1]
    nz = m*n
    nbatch = x.shape[0]

    similarity_map = x.permute(0, 2, 1) @ y # shape: B x m x n
    s = similarity_map.sum(dim=2) # shape: B x m
    d = similarity_map.sum(dim=1) # shape: B x n

    l2_strength=0.0001
    Q = (l2_strength * torch.eye(nz).float()).cuda().unsqueeze(0).repeat(nbatch, 1, 1)
    p = similarity_map.reshape(-1, nz).cuda()
    G = torch.eye(nz).float().unsqueeze(0).repeat(nbatch, 1, 1).cuda() * -1
    h = torch.zeros((nbatch, nz)).cuda()
    A = torch.zeros((nbatch, m+n, nz)).cuda()

    for i in range(m):
        A[:, i, i*n: (i+1)*n] = 1.0

    for i in range(0, m*n, n):
        A[:, m:, i: i+n] = torch.eye(n).cuda()

    b = torch.cat([s, d], dim=1)

    flow = QPFunction(verbose=-1)(Q, p, G, h, A, b)

    input ('flow: {}'.format(flow.shape))


    _, flow = emd_inference_qpth(similarity_map, s, d)
    print (flow.shape)
    input ('check')

    dist = EMD_similarity(x, y)
    print ('shape of dist: ', dist.shape)

    z = torch.zeros_like(y)
    for i in range(y.shape[2]):
        for j in range(y.shape[3]):
            z[:,:,i,j] = y[:,:,y.shape[2]-1-i,y.shape[3]-1-j]

    # print (z)
    print ('z shape: ', z.shape)
    dist = EMD_similarity(x, z)
    print ('shape of dist: ', dist)

    # dist = EMD_similarity(x, x)
    # print ('shape of dist: ', dist)
