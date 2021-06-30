import torch as th
import numpy as np
from scipy.special import erf


def New_sort_sumtest(alpha, M, limit, bar, g, test_idx):
    '''
    New sort method
    :param alpha: an int as the threshold in cutting too large element
    :param M: M is typically the original random walk M
    :param limit: limit is typically the args.num_node
    :param bar: an int used to set the threshold of degree that can be chosen to attack
    :param g: the graph, used to calculate the out_degree of an node
    :return: a list contains the indexs of nodes that needed to be attacked.
    '''
    # print("test_idx len", test_idx.shape)
    # print("test_idx", test_idx)
    # print("test_idx max", max(test_idx))

    test_bin_array = np.zeros((M.shape[0],1))
    test_bin_array[test_idx] = 1
    # print("test_idx_b", test_bin_array.astype(bool))

    s = np.zeros((M.shape[0],1)) # zero vector
    res = [] # res vector

    # make those i has larger degree to -inf
    for i in range(M.shape[0]): 
        if g.out_degree(i) > bar:
            M[:,i] = -float("inf")
    
    # debug
    # print("New_sort(debug): alpha = ", alpha)

    # Greedyly choose the point
    for _ in range(limit):
        L = np.minimum(s+M, alpha)
        L = L.sum(axis=0, where=test_bin_array.astype(bool))
        i = np.argmax(L)
        res.append(i)
        s = s + M[:,i].reshape(M.shape[0],1)
        M[:,i] = -float("inf")
        # delete neighbour
        for neighbor in g.out_edges(i)[1]:
            M[:,neighbor] = -float("inf")
    return res

def New_sort_erf_testsum(sigma, M, limit, bar, g, test_idx):
    '''
    New sort method
    :param alpha: an int as the threshold in cutting too large element
    :param M: M is typically the original random walk M
    :param limit: limit is typically the args.num_node
    :param bar: an int used to set the threshold of degree that can be chosen to attack
    :param g: the graph, used to calculate the out_degree of an node
    :return: a list contains the indexs of nodes that needed to be attacked.
    '''
    # print("test_idx len", test_idx.shape)
    # print("test_idx", test_idx)
    # print("test_idx max", max(test_idx))

    test_bin_array = np.zeros((M.shape[0],1))
    test_bin_array[test_idx] = 1
    # print("test_idx_b", test_bin_array.astype(bool))


    s = np.zeros((M.shape[0],1)) # zero vector
    res = [] # res vector

    # make those i has larger degree to -inf
    for i in range(M.shape[0]): 
        if g.out_degree(i) > bar:
            M[:,i] = -float("inf")
    
    # debug
    # print("New_sort(debug): sigma = ", sigma)

    # Greedyly choose the point
    for _ in range(limit):
        L = erf((s+M)/(sigma*(2**0.5)))
        L = L.sum(axis=0, where=test_bin_array.astype(bool))
        i = np.argmax(L)
        res.append(i)
        s = s + M[:,i].reshape(M.shape[0],1)
        M[:,i] = -float("inf")
        # delete neighbour
        for neighbor in g.out_edges(i)[1]:
            M[:,neighbor] = -float("inf")
    return res



def getScore(K, data):
    Random = data.Prob
    for i in range(K - 1):
        Random = th.sparse.mm(Random, data.Prob)
    return Random.sum(dim=0)


def getScoreGreedy(K, data, bar, num, beta):
    Random = data.Prob
    for i in range(K - 1):
        Random = th.sparse.mm(Random, data.Prob)
    W = th.zeros(data.size, data.size)
    for i in range(data.size):
        value, index = th.topk(Random[i], beta)
        for j, ind in zip(value, index):
            if j != 0:
                W[i, ind] = 1
    SCORE = W.sum(dim=0)
    ind = []
    l = [i for i in range(data.size) if data.g.out_degree(i) <= bar]
    for _ in range(num):
        cand = [(SCORE[i], i) for i in l]
        best = max(cand)[1]
        for neighbor in data.g.out_edges(best)[1]:
            if neighbor in l:
                l.remove(neighbor)
        ind.append(best)
        for i in l:
            W[:, i] -= (W[:, best] > 0) * 1.0
        SCORE = th.sum(W > 0, dim=0)
    return np.array(ind)


def getThrehold(g, size, threshold, num):
    degree = g.out_degrees(range(size))
    Cand_degree = sorted([(degree[i], i) for i in range(size)], reverse=True)
    threshold = int(size * threshold)
    bar, _ = Cand_degree[threshold]
    Baseline_Degree = []
    index = [j for i, j in Cand_degree if i == bar]
    if len(index) >= num:
        Baseline_Degree = np.array(index)[np.random.choice(len(index),
                                                           num,
                                                           replace=False)]
    else:
        while 1:
            bar -= 1
            index_ = [j for i, j in Cand_degree if i == bar]
            if len(index) + len(index_) >= num:
                break
            for i in index_:
                index.append(i)
        for i in np.array(index_)[np.random.choice(len(index_),
                                                   num - len(index),
                                                   replace=False)]:
            index.append(i)
        Baseline_Degree = np.array(index)
    random = [j for i, j in Cand_degree if i <= bar]
    Baseline_Random = np.array(random)[np.random.choice(len(random),
                                                        num,
                                                        replace=False)]
    return bar, Baseline_Degree, Baseline_Random


def getIndex(g, Cand, bar, num):
    ind = []
    for j, i in Cand:
        if g.out_degree(i) <= bar:
            ind.append(i)
        if len(ind) == num:
            break
    return np.array(ind)

def New_sort(alpha, M, limit, bar, g):
    '''
    New sort method
    :param alpha: an int as the threshold in cutting too large element
    :param M: M is typically the original random walk M
    :param limit: limit is typically the args.num_node
    :param bar: an int used to set the threshold of degree that can be chosen to attack
    :param g: the graph, used to calculate the out_degree of an node
    :return: a list contains the indexs of nodes that needed to be attacked.
    '''
    s = np.zeros((M.shape[0],1)) # zero vector
    res = [] # res vector

    # make those i has larger degree to -inf
    for i in range(M.shape[0]): 
        if g.out_degree(i) > bar:
            M[:,i] = -float("inf")
    
    # debug
    # print("New_sort(debug): alpha = ", alpha)

    # Greedyly choose the point
    for _ in range(limit):
        L = np.minimum(s+M, alpha)
        L = L.sum(axis=0)
        i = np.argmax(L)
        res.append(i)
        s = s + M[:,i].reshape(M.shape[0],1)
        M[:,i] = -float("inf")
        # delete neighbour
        for neighbor in g.out_edges(i)[1]:
            M[:,neighbor] = -float("inf")
    return res

def New_sort_erf(sigma, M, limit, bar, g):
    '''
    New sort method
    :param alpha: an int as the threshold in cutting too large element
    :param M: M is typically the original random walk M
    :param limit: limit is typically the args.num_node
    :param bar: an int used to set the threshold of degree that can be chosen to attack
    :param g: the graph, used to calculate the out_degree of an node
    :return: a list contains the indexs of nodes that needed to be attacked.
    '''
    s = np.zeros((M.shape[0],1)) # zero vector
    res = [] # res vector

    # make those i has larger degree to -inf
    for i in range(M.shape[0]): 
        if g.out_degree(i) > bar:
            M[:,i] = -float("inf")
    
    # debug
    # print("New_sort(debug): sigma = ", sigma)

    # Greedyly choose the point
    for _ in range(limit):
        L = erf((s+M)/(sigma*(2**0.5)))
        L = L.sum(axis=0)
        i = np.argmax(L)
        res.append(i)
        s = s + M[:,i].reshape(M.shape[0],1)
        M[:,i] = -float("inf")
        # delete neighbour
        for neighbor in g.out_edges(i)[1]:
            M[:,neighbor] = -float("inf")
    return res

def getM(K, data):
    '''
    Nearly the same as function getScore. Return the random walk matrix directly rather than calculate the col sum.
    '''
    Random = data.Prob
    for i in range(K - 1):
        Random = th.sparse.mm(Random, data.Prob)
    return Random