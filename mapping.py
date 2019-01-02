import networkx as nx
import matplotlib.pyplot as plt
import random
from ai import AStar

class MapRepresentation:

    def __init__(self, random_obstructions=0):

        """
        Generate the map representation in which the robot is traversing. The
        map follows the mathematical definition of a graph i.e., a set of
        vertices (nodes) and edges.
        """

        if random_obstructions < 0:
            random_obstructions = 0
        if random_obstructions > 1:
            random_obstructions = 1

        print('random_obstructions:', random_obstructions)

        self.graph = nx.Graph()   
        
        # Initial map representation.
        self.graph.add_node((0, 0), is_open=True, g_cost=0, h_cost=0)
        self.__init_graph(2, self.graph, list(self.graph.nodes())[0],
            random_obstructions=random_obstructions)

    def __init_graph(self, depth, graph, node, random_obstructions=0):

        """
        Initialize the graph, starting from the root node (0, 0). The graph is
        generated based on the depth. A graph of depth d has 4^d nodes. The
        depth determines how large the initial map representation is.
        """

        if depth < 0:
            return

        # Generate the adjacent nodes at node (x, y). Each node has four
        # neighbors (up, down, left, and, right).
        x, y = node
        for i in range(-1, 2):
            for j in range(-1, 2):
                if abs(i) != abs(j):

                    is_open = True
                    if random.random() < random_obstructions:
                        is_open = False

                    graph.add_node((x + i, y + j), is_open=is_open,
                        h_cost=0, g_cost=0)

                    graph.add_edge((x, y), (x + i, y + j))

        for adj_node in list(graph[x, y]):
            self.__init_graph(depth - 1, self.graph, adj_node,
                random_obstructions=random_obstructions)

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

    def manhattan_distance(n, m):

        """
        Returns the manhattan distance between two nodes n and m.
        """

        nx, ny = n
        mx, my = m

        return abs(nx - mx) + abs(ny - my)

class MapSolver:
    pass
