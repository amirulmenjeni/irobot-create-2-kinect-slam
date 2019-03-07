class Node:

    def __init__(self, label):

        self.label = label
        self.g_cost = 0
        self.h_cost = 0
        self.parent = None

    def __lt__(self, rhs):
        return self.f_cost() < rhs.f_cost()

    def f_cost(self):
        return self.g_cost + self.h_cost
