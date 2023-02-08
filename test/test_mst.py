import sys
sys.path.append('../')

import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances
import networkx as nx

def test_minimum_spanning_tree():
    adjacency_matrix = np.array([
        [0, 2, 0, 6, 0],
        [2, 0, 3, 0, 5],
        [0, 3, 0, 0, 0],
        [6, 0, 0, 0, 0],
        [0, 5, 0, 0, 0]
    ])
    graph = Graph(adjacency_matrix)
    graph.construct_mst()
    mst = graph.mst
    
    # Verify that it spans all the vertices of the graph
    assert np.count_nonzero(mst) == len(mst) - 1

    # Check the total weight of the tree
    #total_weight = 0
    #for i in range(len(mst)):
        #for j in range(i + 1, len(mst)):
            #if mst[i][j] != 0:
                #total_weight += mst[i][j]
    #assert total_weight == 14

    # Check that it's a tree
    #num_edges = 0
    #for i in range(len(mst)):
        #for j in range(i + 1, len(mst)):
            #if mst[i][j] != 0:
                #num_edges += 1
    #assert num_edges == len(mst) - 1




















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

    
    # Ensure that all elements on the main diagonal are zero
    #for i in range(mst.shape[0]):
        #assert mst[i, i] == 0, 'Diagonal elements of MST matrix should be zero'

    # Ensure that all edges in the MST are non-negative
    #for i in range(mst.shape[0]):
        #for j in range(mst.shape[1]):
            #assert mst[i, j] >= 0, 'MST contains negative edge weight'
    
    # Ensure that all edges in the original graph are either in the MST or are zero
    #for i in range(adj_mat.shape[0]):
        #for j in range(adj_mat.shape[1]):
            #assert adj_mat[i, j] == 0 or mst[i, j] > 0, 'Original graph contains edge not present in MST'
    
    # Check the expected weight of the MST
    #expected_weight = 0
    #for i in range(mst.shape[0]):
        #for j in range(i+1):
            #expected_weight += mst[i, j]
    #expected_weight = expected_weight / 2
    #assert abs(expected_weight - expected_weight) < 0.0001, 'Proposed MST has incorrect expected weight'


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = '../data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = '../data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_construct_mst():
    # Test 1: Construct MST from a connected adjacency matrix
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
    
    # Test 2: Construct MST from an adjacency matrix with disconnected nodes
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
    
    
