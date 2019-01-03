import networkx as nx
import matplotlib.pyplot as plt
import random
from ai import AStar

class MapRepresentation:

    def __init__(self, random_obstruction=0):

        """
        Generate the map representation in which the robot is traversing. The
        map follows the mathematical definition of a graph i.e., a set of
        vertices (nodes) and edges.
        """

        if random_obstruction < 0:
            random_obstruction = 0
        if random_obstruction > 1:
            random_obstruction = 1

        self.graph = nx.Graph()   
        
        # Initial map representation.
        self.graph.add_node((0, 0), is_open=True, g_cost=0, h_cost=0)

        self.expand_at(list(self.graph.nodes())[0], depth=2,
            random_obstruction=random_obstruction)

    def expand_at(self, node, depth=1, random_obstruction=0):

        """
        Expand the graph starting from the given node. The depth determines the
        recursive depth of expansion. At most, 4^d new nodes is created upon
        expansion (existing nodes are silently ignored by networkx).
        """

        if depth < 0:
            return

        x, y = node
        for i in range(-1, 2):
            for j in range(-1, 2):
                if abs(i) != abs(j):

                    is_open = True
                    if random.random() < random_obstruction:
                        is_open = False

                    self.graph.add_node((x + i, y + j), is_open=is_open,
                        g_cost=0, h_cost=0)
                    self.graph.add_edge((x, y), (x + i, y + j), weight=1)

        for adj_node in list(self.graph[x, y]):
            self.expand_at(adj_node, depth - 1,
                random_obstruction=random_obstruction)

    def show_graph(self, start=(), goal=()):

        """
        Show the visual representation of the graph.
        """

        solution = []
        if len(start) == 2 and len(goal) == 2:
            solution = AStar.solve(self.graph, start, goal)

        # Color mapping.
        color_map = []
        for node in self.graph.nodes:

            if node in solution:
                color_map.append('green')
                continue

            if self.graph.nodes[node]['is_open']:
                color_map.append('blue')
            else:
                color_map.append('red')

        nx.draw_spectral(self.graph, node_color=color_map, with_labels=True)

        plt.show()
