"""
This blog talk about Constraint Satisfaction Problems (CSP).
One of the most classics optimization problems. To many people 
know it as Back Tracking, or some times Brute Force, but all of them 
have a lot of problems to improve their implementations when this 
improves already have studied.  
"""

import time
import numpy as np
from abc import abstractmethod, ABC

# Some of the most important tools for a developer are maths, algorithms
# and to know several classics problems. But sometimes they aren't enough.
# Sometimes we have a NP-hard problem and we don't know a heuristic to get
# some semi optimal solution.

#  In that case, we usually find a brute force solution and usually it's to
# try with all of feasible solutions. But those problems don't name NP-hard
# for nothing, those problems usually have a very big space of feasible solution (search space).
# So, how our brute force solution is very slow, we start to find some optimization,
# some prunes to skip some of element in the search space.

# But, a lot of ideas that we thing already have discovered and documented.

# ## Constraint Satisfaction Problems (CSP)

# Although, one of main ideas behind CSP is the backtracking search, this new name
# is because csp proposes modeling the problem as an optimization problem, when we have
# some variables and some constraint between these variables. CSP describes some heuristic
# to search the optimal value that they don't take information about the problem's characteristics,
# it build a new problem from the main problem. For all of these, CSP is an uninformed search algorithm
# because it doesn't take more information from the problem than the number of variables and the constraints

class CspBase(ABC):

    # With a little information, we can build a graph, when each variable is a node and two nodes
    # are linked if there's a constraint between them. Each node should have its domain, a list with
    # all of possible values for this node. Some problems have a global domain for all of variables,
    # but it's a little implementation change.

    def __init__(self, variables, adj_list, domains, verbose=True) -> None:
        self.N = len(variables)  # number of node
        self.variables = variables  # nodes's name
        self.assignation = [None] * self.N  # nodes' values
        self.adj_list = adj_list  # constraint between them
        self.domains = domains  # nodes' domains
        self.verbose = verbose
        self.var_map = dict([(v, i) for i, v in enumerate(variables)])

    # The graph theory is a powerful tools to solve problems, there're to many problems that they
    # have a easy solution by a graph modeling. In this case, CSP use this modeling to suggest some
    # heuristics to visit this graph and to find optimal solutions

    # ## Backtracking Search Base

    # How we have already said behind CSP to be the Backtracking Search.
    # So, the main method of our class is the function `_search`.
    # It is an high level description of our search mechanic. 

    def _search(self, get_first):
        if self.is_finished: # we need a way to know when we have a full possible solution
            return self.callback() # we need some instruction to follow after get a possible solution

        # This is the CSP's first proposition.
        # We will assign one variable(node) in each step.
        # There're other approaches, like constructive or statistical,
        # that they try to assign more variables in each step, but 
        # they usually don't visit all the space and some time 
        # they find an optimal local solution and don't visit the best solution  
        var = self.select_unassigned_variable()

        # After to choice what is the next variable, we have to iterate 
        # each values in the domain of chosen variable.
        # Sometimes is very important to iterate this domain in specific order
        for value in self.order_domain_values(var):

            # save domains because inference step can update them by the last assignation
            save_domains = self.domains
            self.assign(var, value) # assign the next value to the chosen variable 

            # The most important ideas of CSP will be here, at the inference time, 
            # after to assign the chosen variable with the next value.
            # Which each assign the possibles values for the other unassigned variables may reduce.
            # While in bark tracking search we maintain all of these unfeasible values into the domains,
            # hence there will be a lot of unfeasible final solutions.
            # Now, we will remove some unfeasible values from the domains of unassigned variables.    
            if self.inference(var):  # If we get a False at the inference time it mean that the previous assign are bad
                result = self._search(get_first)
                if get_first and result:
                    return result

            self.domains = save_domains # We recovered the previous domains 

        # Who this search is a recursive process, 
        # so is very important to remove values from the chosen variable 
        self.unassign(var) 
        return None



    # ## Problem Interface

    # How we have already said CSP is uninformed search algorithm.
    # So, it only need to know about the structure of the problems and 
    # few others details about it, por example how evaluate each restriction.

    # For that, we define an interface which the user can describe some 
    # characteristic about the problem. The main method into this interface is 
    # the function `check_constraint`, 
    # because the other function into this interface use it to make a basic implementation.
    # Usually we can use more information about the problem to implement this other function 
    # as an efficiently way   

    @abstractmethod
    def check_constraint(self, a, value_a, b, value_b) -> bool:
        """
            This function should return a boolean result to evaluate 
            the constraint between the nodes a and b with the values 
            value_a and value_b respectively 
        """

    def update_domain(self, var, unassign_var):
        """
            This function should update all of domain of unassigned variables, 
            If var is last assigned variable.  
        """

        return [val for val in self.domains[unassign_var]
                if self.check_constraint(var, self.assignation[var], unassign_var, val)]

    def n_conflicts(self, var, value):
        """
            This function should return how much constraint we could fail if 
            we assign the variable var with value. It mean to test each var's constraint
            with each possible value of the other variable  
        """

        return sum(
            [not self.check_constraint(var, value, other, self.assignation[other])
             for other in self.adj_list[var] if not self.assignation[other] is None]
        )

    def goal_test(self) -> bool:
        """
            This function should check if the finished assignment is valid  
        """
        return self._goal_test()

    def _goal_test(self) -> bool:
        for i in range(self.N):
            for j in filter(lambda x: x != i, range(self.N)):
                if not self.check_constraint(i, self.assignation[i], j, self.assignation[j]):
                    return False


    # ## Search Interface

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

    def __setitem__(self, var, value):
        index = self.var_map[var]
        self.explored_state = 0
        self.assign(index, value)

    def assign(self, var, value):
        self.assignation[var] = value
        self.explored_state += 1

    def unassign(self, var):
        self.assignation[var] = None

    def inference(self, var):
        return True






    



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

    def structure_optimize_and_search(self):
        # BUG: When there are more than one bridge in the same ccc

        start = time.time()

        ccc, cc_distribution, bridges = self.find_bridges()
        flatten_bridges = set([x for edges in bridges for x in edges])

        if len(bridges) > 1:
            adj_bridges = dict([(edge[0 + i], edge[1 - i])
                                for edge in bridges for i in range(2)])

            visited = dict([(x, -1) for x in adj_bridges.keys()])
            ccc_b = 0
            for x in filter(lambda x: visited[x] == -1,  adj_bridges.keys()):
                visited = self.bfs(x, ccc_b, visited, adj_bridges)
                ccc_b += 1

            group_by_ccc = dict((value, key) for key, value in visited.items())
            bridges = list(group_by_ccc.values())

        set_ccc_bridges = set([cc_distribution[b[0]] for b in bridges])
        if ccc > 1:
            index = 0
            while index < len(cc_distribution) and len(set_ccc_bridges) < ccc:
                cc = cc_distribution[index]
                if not cc in set_ccc_bridges:
                    bridges.append([index])
                    set_ccc_bridges.add(index)
                index += 1

        problems_collection = []
        for sub_problem in bridges:
            sub_collection = []

            if len(sub_problem) == 1:

                # connected complement case
                sub_collection.append(
                    new_problem=self.reduce_problem(
                        self.find_sub_problem(sub_problem[0])
                    )
                )

            else:

                # bridges decomposition case
                sub_collection.append(self.reduce_problem(sub_problem))

                for n in sub_problem:
                    new_problem = self.reduce_problem(
                        self.find_sub_problem(n, flatten_bridges)
                    )
                    sub_collection.append((n, new_problem))

            problems_collection.append(sub_collection)

        solutions = [{}]
        for tree in problems_collection:
            main_problem: CspBase = tree.pop(0)
            main_result = main_problem.search()
            if len(tree) > 0:
                result = []
                for main_values in main_result:
                    temp_result = [{}]
                    for problem, root_value in zip(tree, main_values):
                        root, problem = problem
                        problem[root] = root_value

                        new_r = [
                            dict(
                                [(n, v) for n, v in zip(problem.variables, r)]
                            )
                            for r in problem.search()]
                        temp_result = [
                            x | y for x in temp_result for y in new_r]

                    result += temp_result
            else:

                result = [
                    dict([(n, v) for n, v in zip(main_problem.variables, r)])
                    for r in main_result
                ]

            solutions = [x | y for x in solutions for y in result]

        end = time.time() - start

        if self.verbose:
            print(f"""
            Csp Search Result:
                results: {len(solutions)}
                time: {end}s
                n problems: {len([n for ps in problems_collection for n in ps]) + len(problems_collection)}
                connected complements count: {ccc}
                cc distribution: {cc_distribution}
                bridges: {bridges}
            """)

        return [[s[i] for i in range(self.N)] for s in solutions]

    def get_connected_complement(self):
        visited = [-1] * self.N

        ccc = 0  # connected complement counter
        for n in range(self.N):
            if visited[n] == -1:
                visited = self.bfs(n, ccc, visited, self.adj_list)
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

    def reduce_problem(self, nodes, verbose=False):
        nodes_map = dict([(n, i) for i, n in enumerate(nodes)])
        domains = [self.domains[n] for n in nodes]
        adj_list = [
            [nodes_map[adj] for adj in self.adj_list[n] if adj in nodes_map]
            for n in nodes
        ]

        return type.__call__(self.__class__, nodes, adj_list, domains, verbose=verbose)

    def find_sub_problem(self, root, exclude=[]):
        Q = [root]
        s = set([root])
        while len(Q) != 0:
            v = Q.pop(0)
            adjs = filter(
                lambda x: not x in exclude and not x in s,
                self.adj_list[v]
            )

            for w in adjs:
                Q.append(w)
                s.add(w)

        return sorted(s)
        # Helpers

    def _clone_domain_if(self, condition):
        if condition:
            return [self.domains[i] for i in range(self.N)]
        else:
            return self.domains

    def unassign_adj(self, var):
        return [x for x in self.adj_list[var] if self.assignation[x] is None]



    def bfs(self, root, mark, visited, adj_list):
        Q = [root]
        while len(Q) != 0:
            v = Q.pop(0)
            for adj in filter(lambda x: visited[x] == -1, adj_list[v]):
                visited[adj] = mark
                Q.append(adj)

        return visited


# ## Complement Read
# https://gki.informatik.uni-freiburg.de/teaching/ss14/gki/lectures/ai05.pdf
