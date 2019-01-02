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
        
        for adj_node in list(graph[node]):
            if graph.nodes[adj_node][attr_key] == open_value:
                yield adj_node

class AStar:

    def __find_node_in_queue(queue, node):

        for n in queue.queue:
            if n[1] == node:
                return n
        return None

    def __find_greater_f_in_queue(queue, node, node_f_cost):

        for n in queue.queue:
            if n[1] == node and n[0] > node_f_cost:
                n[0] = node_f_cost
        return None

    def __cleanup(graph, explored_nodes):

        for node in explored_nodes:
            self.graph.nodes[node]['g_cost'] = 0
            self.graph.nodes[node]['h_cost'] = 0

    def solve(graph, start, goal):

        """

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

            print(node, f, graph.nodes[node])

            # Flag this node as explored.
            explored_nodes[node] = True

            # Goal test.
            if node == goal:
                print('Solution found!')
                solution_list = [node]
                parent = solution.nodes[node]['parent']
                print('%s <- %s' % (parent, node))
                node = parent

                while node is not None:
                    solution_list.append(node)
                    parent = solution.nodes[node]['parent']
                    # print('%s <- %s' % (parent, node))
                    node = parent

                print('Exiting')
                return solution_list[::-1] # Reverse the list.

            # Iterate through this node's neighbouring nodes.
            for neighbour in Util.adj_open_nodes(graph, node):

                g_cost = search_graph.nodes[node]['g_cost'] + 1
                h_cost = Util.manhattan_distance(neighbour, goal)
                f_cost = g_cost + h_cost

                if neighbour not in explored_nodes:
                    print('Parent of %s: %s' % (neighbour, node))
                    solution.add_node(neighbour, parent=node)

                if neighbour not in explored_nodes and\
                   AStar.__find_node_in_queue(frontier_queue, neighbour)\
                    is None:

                    # Update g-cost and h-cost of this neighbouring node.
                    search_graph.add_node(neighbour, g_cost=0, h_cost=0)
                    search_graph.nodes[neighbour]['g_cost'] = g_cost
                    search_graph.nodes[neighbour]['h_cost'] = h_cost

                    frontier_queue.put((f_cost, neighbour))
                
                else:
                    # If the neighbour is in the frontier_queue with a higher
                    # f-cost, then replace that frontier node with neighbour.
                    AStar.__find_greater_f_in_queue(frontier_queue,
                        neighbour, f_cost)

        # No solution found.
        return []

