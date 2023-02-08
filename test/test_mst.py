#modules used
import sys
sys.path.append('../')

import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances
import networkx as nx

###################
def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    ### additional assertions ###

    #1: Ensure that all elements on the main diagonal are zero
    for i in range(mst.shape[0]):
        assert mst[i, i] == 0, 'Diagonal elements of MST matrix should be zero'

    #2: Ensure that all edges in the MST are non-negative
    for i in range(mst.shape[0]):
        for j in range(mst.shape[1]):
            assert mst[i, j] >= 0, 'MST contains negative edge weight'
    
    
    #3: Check that the MST is connected by ensuring that there is a path between any two vertices in the graph
    def is_connected(adj_mat):
        n = adj_mat.shape[0]
        visited = [False] * n
        queue = [0]
        visited[0] = True
        while queue:
            u = queue.pop(0)
            for v in range(n):
                if adj_mat[u][v] and not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return all(visited)

    assert is_connected(mst), 'MST is not connected'
    
    #4: Check that the number of edges in the MST is equal to the number of vertices minus 1, i.e.
    n = adj_mat.shape[0]
    num_edges = np.count_nonzero(mst) // 2
    assert num_edges == n - 1, 'Incorrect number of edges in MST'



def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.
    https://bioconductor.org/packages/release/bioc/html/slingshot.html
    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_construct_mst():
    # Test 1: can my function construct an MST from a connected adjacency matrix
    # testing via direct comparison with nx MST output
    adjacency_mat = np.array([[0, 2, 0, 6, 0],
                              [2, 0, 3, 8, 5],
                              [0, 3, 0, 0, 7],
                              [6, 8, 0, 0, 9],
                              [0, 5, 7, 9, 0]])
    graph = Graph(adjacency_mat)
    graph.construct_mst()
    expected_mst = nx.minimum_spanning_tree(nx.from_numpy_array(adjacency_mat))
    expected_mst = nx.to_numpy_array(expected_mst)
    np.testing.assert_array_equal(graph.mst, expected_mst)
    
    # Test 2: can my function construct an MAST from an adjacency matrix with disconnected nodes
    # testing via direct comparison with nx MST output
    adjacency_mat = np.array([[0, 2, 0, 6, 0],
                              [2, 0, 3, 8, 5],
                              [0, 3, 0, 0, 7],
                              [6, 8, 0, 0, 9],
                              [0, 5, 7, 9, 0]])
    graph = Graph(adjacency_mat)
    graph.construct_mst()
    expected_mst = nx.minimum_spanning_tree(nx.from_numpy_array(adjacency_mat))
    expected_mst = nx.to_numpy_array(expected_mst)
    np.testing.assert_array_equal(graph.mst, expected_mst)
    
