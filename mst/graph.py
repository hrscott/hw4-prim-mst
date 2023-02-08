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

    def construct_mst(self):
        n = len(self.adj_mat)
        mst = np.zeros((n, n))
        visited = [False] * n
        min_edges = [(float('inf'), 0, 0) for i in range(n)]
        heap = []
        
        # Start the MST with vertex 0
        heapq.heappush(heap, (0, 0, -1))
        
        while heap:
            weight, u, v = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            if v != -1:
                mst[u][v] = weight
                mst[v][u] = weight
            for i in range(n):
                if self.adj_mat[u][i] > 0 and not visited[i]:
                    heapq.heappush(heap, (self.adj_mat[u][i], i, u))
        self.mst = mst

    #some helper  helper functions for visualization
    def visualize_mst(self):
        graph = nx.Graph(mst)
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True)
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
        plt.show()
    
    def visualize_graph(self):
        G = nx.Graph(self.adj_mat)
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
        plt.show()
    
  
     


     