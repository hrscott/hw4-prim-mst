import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        This Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

### Prim's Algorithm Implementation
    def construct_mst(self):
        # number of vertices in the graph
        n = len(self.adj_mat)
        # initialize the minimum spanning tree adjacency matrix
        mst = np.zeros((n, n))
        # keep track of which vertices have been visited
        visited = [False] * n
        # priority queue for storing the minimum edge weights
        heap = []
        
        # Start the MST with vertex 0
        heapq.heappush(heap, (0, 0, -1))
        
        # keep expanding the MST until all vertices have been visited
        while heap:
        # get the edge with the minimum weight
            weight, u, v = heapq.heappop(heap)
        # if the destination vertex has already been visited, skip this edge
            if visited[u]:
                continue
        # mark the destination vertex as visited
            visited[u] = True
        # add this edge to the minimum spanning tree
            if v != -1:
                mst[u][v] = weight
                mst[v][u] = weight
        # add all the edges connected to the destination vertex to the priority queue
            for i in range(n):
                if self.adj_mat[u][i] > 0 and not visited[i]:
                    heapq.heappush(heap, (self.adj_mat[u][i], i, u))
        # store the minimum spanning tree in the class instance
        self.mst = mst

    
    #some helper  helper functions for visualization

    def visualize_graph(self):
        graph = nx.from_numpy_array(self.adj_mat)
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True)
        labels = nx.get_edge_attributes(graph,'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
        plt.show()
  
     
    def visualize_mst(self):
        mst_graph = nx.from_numpy_array(self.mst)
        pos = nx.spring_layout(mst_graph)
        nx.draw(mst_graph, pos, with_labels=True)
        labels = nx.get_edge_attributes(mst_graph,'weight')
        nx.draw_networkx_edge_labels(mst_graph, pos, edge_labels=labels)
        plt.show()

     
  
     


     
