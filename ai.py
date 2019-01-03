import queue
import networkx as nx

class Util:

    def manhattan_distance(n, m):

        """
        Returns the manhattan distance between nodes n and m.

        @param n: 2-tuple.
        @param m: 2-tuple.
        """

        return abs(n[0] - m[0]) + abs(n[1] - m[1])

    def adj_open_nodes(graph, node, attr_key='is_open', open_value=True):

        """
        Generate the accessible adjacent nodes from the given node in the given
        graph.
        """
        
        for adj_node in list(graph[node]):
            if graph.nodes[adj_node][attr_key] == open_value:
                yield adj_node

class AStar:

    def __find_node_in_queue(queue, node):

        """
        Find the given node in the queue. If found, the node is returned.
        Otherwise, None is returned.
        """

        for n in queue.queue:
            if n[1] == node:
                return n
        return None

    def solve(graph, start, goal):

        """
        Returns a list of nodes that represents the solution or path from the
        starting node to the goal node on the given graph. Makes use of A*
        search algorithm.
        """

        # Frontier queue, containing 2-tuples: (priority, (x, y)).
        frontier_queue = queue.PriorityQueue()
        g_cost = 0
        h_cost = Util.manhattan_distance(start, goal)
        frontier_queue.put((g_cost + h_cost, start))

        explored_nodes = {}
        explored_nodes[start] = True

        solution = nx.Graph()
        solution.add_node(start, parent=None)

        search_graph = nx.Graph()
        search_graph.add_node(start, g_cost=0, h_cost=h_cost)

        while not frontier_queue.empty():

            # Get the highest node with highest priority (lowest f-cost).
            f, node = frontier_queue.get()

            # Flag this node as explored.
            explored_nodes[node] = True

            # Goal test.
            if node == goal:
                solution_list = [node]
                node = solution.nodes[node]['parent']

                while node is not None:
                    solution_list.append(node)
                    node = solution.nodes[node]['parent']

                return solution_list[::-1] # Reverse the list.

            # Iterate through this node's neighbouring nodes.
            for neighbour in Util.adj_open_nodes(graph, node):

                g_cost = search_graph.nodes[node]['g_cost'] +\
                    graph.edges[node, neighbour]['weight']
                h_cost = Util.manhattan_distance(neighbour, goal)
                f_cost = g_cost + h_cost

                if neighbour not in explored_nodes:
                    solution.add_node(neighbour, parent=node)

                if neighbour not in explored_nodes and\
                   AStar.__find_node_in_queue(frontier_queue, neighbour)\
                    is None:

                    # Update g-cost and h-cost of this neighbouring node.
                    search_graph.add_node(neighbour, g_cost=0, h_cost=0)
                    search_graph.nodes[neighbour]['g_cost'] = g_cost
                    search_graph.nodes[neighbour]['h_cost'] = h_cost

                    frontier_queue.put((f_cost, neighbour))
                
        # No solution found.
        return []

