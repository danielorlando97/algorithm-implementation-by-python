import time
import numpy as np
import itertools as it

# https://gki.informatik.uni-freiburg.de/teaching/ss14/gki/lectures/ai05.pdf


class CspBase:
    def __init__(self, variables, adj_list, domains, verbose=True) -> None:
        self.N = len(variables)
        self.variables = variables
        self.assignation = [None] * self.N
        self.adj_list = adj_list
        self.domains = domains
        self.verbose = verbose

    # Problem Interface

    def check_constraint(self, a, value_a, b, value_b) -> bool:
        pass

    def update_domain(self, var, unassign_var):
        return [val for val in self.domains[unassign_var]
                if self.check_constraint(var, self.assignation[var], unassign_var, val)]

    def n_conflicts(self, var, value):
        return sum(
            [not self.check_constraint(var, value, other, self.assignation[other])
             for other in self.adj_list[var] if not self.assignation[other] is None]
        )

    def goal_test(self) -> bool:
        return self._goal_test()

    # Search Interface

    @ property
    def is_finished(self):
        return all(not x is None for x in self.assignation)

    def callback(self):
        if self.goal_test():
            self.result_list.append([val for val in self.assignation])
            return self.result_list[-1]

    def select_unassigned_variable(self):
        return self.assignation.index(None)

    def order_domain_values(self, var):
        return self.domains[var]

    def assign(self, var, value):
        self.assignation[var] = value
        self.explored_state += 1

    def unassign(self, var):
        self.assignation[var] = None

    def inference(self, var):
        return True

    def search(self, get_first=False):
        self.result_list = []
        self.explored_state = 0

        start = time.time()
        result = self._search(get_first)
        end = time.time() - start

        if self.verbose:
            print(f"""
            Csp Search Result:
                results: {len(self.result_list)}
                time: {end}s
                explored states: {self.explored_state}
            """)

        return result if get_first else self.result_list

    # Variable ordering
    def minimum_remaining_values(self):
        return min(
            [(len(list(self.order_domain_values(var))), var)
             for var, value in enumerate(self.assignation) if value is None]
        )[1]

    # Value ordering and filter

    def least_constraining_values(self, var):
        return sorted(self.domains[var], key=lambda val: self.n_conflicts(var, val))

    def checking_after_assign(self, var):
        return filter(lambda x: self.n_conflicts(var, x) == 0, self.domains[var])

    # Inference Methods

    def forward_checking(self, var):
        """
            for this heuristic always chosen var to assign has the
            specific domains that it is consistence with the assignation.
            ForwardChecking is better heuristic when each assignation only
            can drop one value from the other domains. Because if the constraint
            is one of one then domains with length greater than 2 always are valid
        """

        unassign_adj = self.unassign_adj(var)

        self.domains = self._clone_domain_if(len(unassign_adj) != 0)
        for adj in unassign_adj:
            self.domains[adj] = self.update_domain(var, adj)
            if len(self.domains[adj]) == 0:
                return False

        return True

    def constraint_propagation(self, var):
        Q = [(var, adj) for adj in self.unassign_adj(var)]
        self.domains = self._clone_domain_if(len(Q) != 0)

        while len(Q) != 0:
            xi, xk = Q.pop(0)
            new_k_domain = self.update_domain(xi, xk)

            if len(new_k_domain) == 0:
                return False
            if len(new_k_domain) != len(self.domains[xk]):
                self.domains[xk] = new_k_domain
                for adj_xk in self.unassign_adj(xk):
                    if adj_xk != xi:
                        Q.append((xk, adj_xk))

        return True

    # Structure Optimization

    def get_connected_complement(self):
        visited = [-1] * self.N

        def bfs(root, mark):
            Q = [root]
            while len(Q) != 0:
                v = Q.pop(0)
                for adj in filter(lambda x: visited[x] == -1, self.adj_list[v]):
                    visited[adj] = mark
                    Q.append(adj)

        ccc = 0  # connected complement counter
        for n in range(self.N):
            if visited[n] == -1:
                bfs(n, ccc)
                ccc += 1

        return ccc, visited

    def find_bridges(self):
        visited = [-1] * self.N
        # time input (depth at which node i appears for the first time)
        tin = [-1] * self.N
        # low (length of minimal cycle that it contain the node i)
        low = [-1] * self.N
        timer = 0  # how in a dfs tree there can be no lateral edge, timer is the same than deep
        ccc = 0  # connected complement counter
        bridges = []

        def dfs(n, p=-1):
            nonlocal timer
            visited[n] = ccc
            tin[n] = low[n] = timer  # the less deeper node
            timer += 1

            for adj in self.adj_list[n]:
                if adj == p:
                    continue
                if visited[adj] != -1:
                    low[n] = min(low[n], tin[adj])
                else:
                    dfs(adj, n)
                    low[n] = min(low[n], low[adj])
                    if low[adj] > tin[n]:
                        bridges.append((n, adj))

        for i in range(self.N):
            if visited[i] == -1:
                dfs(i)
                ccc += 1

        return ccc, visited, bridges

    def is_tree(self):
        visited = [False] * self.N
        Q = [0]
        while len(Q) != 0:
            v = Q.pop(0)
            visited[v] = True

            adj_iter = filter(lambda x: x != v, self.adj_list[v])
            for adj in adj_iter:
                if visited[adj]:
                    return False
                Q.append(adj)

        return all(visited)

    # https://courses.cs.duke.edu/spring01/cps271/LECTURES/cspII.pdf
    # https://ktiml.mff.cuni.cz/~bartak/constraints/nosearch.html
    def get_minimal_cut_set(self):
        degrees = [len(self.adj_list[i]) for i in range(self.N)]
        selected = [False] * self.N

        root = np.argmin(degrees)
        selected[root] = True
        pool = [(root, i) for i in self.adj_list[root]]
        adj_tree = [[] for i in range(self.N)]

        for _ in range(self.N - 1):
            i = np.argmin(map(lambda x: degrees[x[0]] + degrees[x[1]], pool))

            xi, xk = pool.pop(i)
            degrees[xi] -= 1
            degrees[xk] -= 1
            adj_tree[xi].append(xk)
            adj_tree[xk].append(xi)
            selected[xk] = True

            pool = [edge for edge in pool if edge[1] != xk] + \
                [(xk, i) for i in self.adj_list[xk] if not selected[i]]

    # Helpers

    def _clone_domain_if(self, condition):
        if condition:
            return [self.domains[i] for i in range(self.N)]
        else:
            return self.domains

    def unassign_adj(self, var):
        return [x for x in self.adj_list[var] if self.assignation[x] is None]

    def _search(self, get_first):
        if self.is_finished:
            return self.callback()

        var = self.select_unassigned_variable()
        for value in self.order_domain_values(var):

            # save domains because inference step can update them
            # by the last assignation
            save_domains = self.domains
            self.assign(var, value)

            if self.inference(var):
                result = self._search(get_first)
                if get_first and result:
                    return result

            self.domains = save_domains

        self.unassign(var)
        return None

    def _goal_test(self) -> bool:
        for i in range(self.N):
            for j in filter(lambda x: x != i, range(self.N)):
                if not self.check_constraint(i, self.assignation[i], j, self.assignation[j]):
                    return False
