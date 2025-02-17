import json
from copy import deepcopy
import itertools
from collections import deque, defaultdict

########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################


class Inference:
    def __init__(self, data):
        """
        Initialize the Inference class with the input data.

        Parameters:
        -----------
        data : dict
            The input data containing the graphical model details, such as variables, cliques, potentials, and k value.

        What to do here:
        ----------------
        - Parse the input data and store necessary attributes (e.g., variables, cliques, potentials, k value).
        - Initialize any data structures required for triangulation, junction tree creation, and message passing.

        Refer to the sample test case for the structure of the input data.
        """

        def adjacency_list(edge_list, nodes):
            adj_list = {node: [] for node in nodes}
            for u, v in edge_list:
                adj_list[u].append(v)
                adj_list[v].append(u)
            return adj_list

        self.clique_potentials = {}
        self.edges = set()
        self.nodes = set()

        self.k = data["k value (in top k)"]

        for i in data["Cliques and Potentials"]:
            if tuple(i["cliques"]) not in self.clique_potentials:
                self.clique_potentials[tuple(i["cliques"])] = i["potentials"]
            else:
                self.clique_potentials[tuple(i["cliques"])] = [
                    self.clique_potentials[tuple(i["cliques"])][j] * i["potentials"][j]
                    for j in range(len(i["potentials"]))
                ]
        for i in self.clique_potentials:
            for j in i:
                for k in i:
                    if j < k:
                        self.edges.add((j, k))
        for i in self.edges:
            for j in i:
                self.nodes.add(j)
        self.nodes = sorted(list(self.nodes))
        self.adjlist = adjacency_list(self.edges, self.nodes)

    def triangulate_and_get_cliques(self):
        """
        Triangulate the undirected graph and extract the maximal cliques.

        What to do here:
        ----------------
        - Implement the triangulation algorithm to make the graph chordal.
        - Extract the maximal cliques from the triangulated graph.
        - Store the cliques for later use in junction tree creation.

        Refer to the problem statement for details on triangulation and clique extraction.
        """

        def is_simplicial(adjlist, node):
            neighbors = adjlist[node]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] not in adjlist[neighbors[i]]:
                        return False
            return True

        def find_simplicial_vertex(adjlist):
            for node in adjlist:
                if is_simplicial(adjlist, node):
                    return node
            return None

        def make_vertex_simplicial(adjlist, node):
            neighbors = adjlist[node]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] not in adjlist[neighbors[i]]:
                        adjlist[neighbors[i]].append(neighbors[j])
                        adjlist[neighbors[j]].append(neighbors[i])
            return adjlist

        def chordal_graph_with_heuristic(adjlist):
            chordal_adjlist = deepcopy(adjlist)
            elimination_order = []
            cliques = []

            while len(elimination_order) < len(adjlist):
                simplicial_node = find_simplicial_vertex(chordal_adjlist)

                if simplicial_node is not None:
                    elimination_order.append(simplicial_node)
                    clique = {simplicial_node}
                    neighbours = chordal_adjlist[simplicial_node]
                    for neigh in neighbours:
                        if all(neigh in chordal_adjlist[i] for i in clique):
                            clique.add(neigh)
                    cliques.append(clique)
                    for neighbor in chordal_adjlist[simplicial_node]:
                        chordal_adjlist[neighbor].remove(simplicial_node)
                    del chordal_adjlist[simplicial_node]
                else:
                    degrees = {
                        node: len(neighbors)
                        for node, neighbors in chordal_adjlist.items()
                    }
                    least_degree_node = min(degrees, key=degrees.get)
                    chordal_adjlist = make_vertex_simplicial(
                        chordal_adjlist, least_degree_node
                    )
                    elimination_order.append(least_degree_node)
                    clique = {least_degree_node}
                    neighbours = chordal_adjlist[least_degree_node]
                    for neigh in neighbours:
                        if all(neigh in chordal_adjlist[i] for i in clique):
                            clique.add(neigh)
                    cliques.append(clique)
                    for neighbor in chordal_adjlist[least_degree_node]:
                        chordal_adjlist[neighbor].remove(least_degree_node)
                    del chordal_adjlist[least_degree_node]

            max_cliques = []
            for clique in cliques:
                is_subset = False
                for m_clique in max_cliques:
                    if clique.issubset(m_clique):
                        is_subset = True
                        break
                if not is_subset:
                    max_cliques.append(clique)

            return max_cliques

        max_cliques = chordal_graph_with_heuristic(self.adjlist)
        self.max_cliques = max_cliques
        # print("Adjacency List:")
        # for node, neighbors in self.adjlist.items():
        #     print(f"{node}: {neighbors}")
        # print("Maximal Cliques:", self.max_cliques)

    def get_junction_tree(self):
        """
        Construct the junction tree from the maximal cliques.

        What to do here:
        ----------------
        - Create a junction tree using the maximal cliques obtained from the triangulated graph.
        - Ensure the junction tree satisfies the running intersection property.
        - Store the junction tree for later use in message passing.

        Refer to the problem statement for details on junction tree construction.
        """

        def create_junction_graph(cliques):
            junction_graph = {}
            for i, clique1 in enumerate(cliques):
                for j, clique2 in enumerate(cliques):
                    if i < j:
                        intersection = clique1.intersection(clique2)
                        if intersection:
                            weight = len(intersection)
                            if tuple(clique1) not in junction_graph:
                                junction_graph[tuple(clique1)] = []
                            if tuple(clique2) not in junction_graph:
                                junction_graph[tuple(clique2)] = []
                            junction_graph[tuple(clique1)].append(
                                (tuple(clique2), weight)
                            )
                            junction_graph[tuple(clique2)].append(
                                (tuple(clique1), weight)
                            )
            return junction_graph

        self.junction_graph = create_junction_graph(self.max_cliques)
        # print("Junction Graph:")
        # for node, neighbors in self.junction_graph.items():
        #     print(f"{node}: {neighbors}")

        def make_junction_tree(junction_graph):
            # Convert the junction graph to a list of edges suitable for Kruskal's algorithm
            edges = []
            for node, neighbors in junction_graph.items():
                for neighbor, weight in neighbors:
                    if (neighbor, node, weight) not in edges:
                        edges.append((node, neighbor, weight))

            # Sort edges by weight in descending order
            edges.sort(key=lambda x: -x[2])

            parent = {}
            rank = {}

            def find(node):
                if parent[node] != node:
                    parent[node] = find(parent[node])
                return parent[node]

            def union(node1, node2):
                root1 = find(node1)
                root2 = find(node2)
                if root1 != root2:
                    if rank[root1] > rank[root2]:
                        parent[root2] = root1
                    else:
                        parent[root1] = root2
                        if rank[root1] == rank[root2]:
                            rank[root2] += 1

            # Initialize the union-find structure
            for node in junction_graph:
                parent[node] = node
                rank[node] = 0

            max_spanning_tree = {}

            for node1, node2, weight in edges:
                if find(node1) != find(node2):
                    union(node1, node2)
                    if node1 not in max_spanning_tree:
                        max_spanning_tree[node1] = []
                    if node2 not in max_spanning_tree:
                        max_spanning_tree[node2] = []
                    max_spanning_tree[node1].append(node2)
                    max_spanning_tree[node2].append(node1)

            return max_spanning_tree

        
        # p = {
        #     "abc": [("cde", 1), ("acf", 2),("agf",1)],
        #     "acf": [("abc", 2), ("cde", 1),("agf",2)],
        #     "cde": [("abc", 1), ("acf", 1)],
        #     "agf": [("abc", 1), ("gh", 1),("acf",2)],
        #     "gh": [("agf", 1)]
        # }

        # print("Junction Tree:")
        # for node, neighbors in self.junction_tree.items():
        #     print(f"{node}: {neighbors}")
        # print(make_junction_tree(p))

        def get_message_passing_order(adj_list, root):

            # Step 1: Convert undirected tree to a directed tree using BFS
            directed_tree = defaultdict(list)
            parent = {root: None}
            queue = deque([root])
            
            while queue:
                node = queue.popleft()
                for neighbor in adj_list[node]:
                    if neighbor != parent.get(node):  # Avoid going back to parent
                        parent[neighbor] = node
                        directed_tree[node].append(neighbor)
                        queue.append(neighbor)
            
            # Step 2: Perform postorder traversal to determine message passing orders
            child_to_parent = []
            parent_to_child = []
            
            def postorder_traversal(node):
                for child in directed_tree[node]:
                    postorder_traversal(child)
                    child_to_parent.append((child, node))
            
            def preorder_traversal(node):
                for child in directed_tree[node]:
                    parent_to_child.append((node, child))
                    preorder_traversal(child)
            
            postorder_traversal(root)
            preorder_traversal(root)
            
            return child_to_parent + parent_to_child
        
        self.junction_tree = make_junction_tree(self.junction_graph)
        self.opt_order = get_message_passing_order(self.junction_tree, list(self.junction_tree.keys())[0])
        print("Opt order: ",self.opt_order)

    def assign_potentials_to_cliques(self):
        """
        Assign potentials to the cliques in the junction tree.

        What to do here:
        ----------------
        - Map the given potentials (from the input data) to the corresponding cliques in the junction tree.
        - Ensure the potentials are correctly associated with the cliques for message passing.

        Refer to the sample test case for how potentials are associated with cliques.
        """

        def multiply_potentials(potential1, potential2, potn1_depends_on, potn2_depends_on):
            potential_new = [None] * (2 ** (len(set(potn1_depends_on).union(potn2_depends_on))))
            for val in itertools.product([0, 1], repeat=len(set(potn1_depends_on).union(potn2_depends_on))):
                assignmt = dict(zip(set(potn1_depends_on).union(potn2_depends_on), val))
                potn1_idx = 0
                for t in potn1_depends_on:
                    potn1_idx = (potn1_idx << 1) + assignmt[t]

                potn2_idx = 0
                for t in potn2_depends_on:
                    potn2_idx = (potn2_idx << 1) + assignmt[t]

                net_idx = 0
                for t in set(potn1_depends_on).union(potn2_depends_on):
                    net_idx = (net_idx << 1) + assignmt[t]

                potential_new[net_idx] = potential1[potn1_idx] * potential2[potn2_idx]

            return potential_new, set(potn1_depends_on).union(potn2_depends_on)

        max_clique_potentials = {}
        visited_cliques = set()
        cliques_assigned = {}
        for max_clique in self.max_cliques:
            subset_cliques = []
            cliques_assigned[tuple(max_clique)] = []
            for clique in self.clique_potentials:
                if set(clique).issubset(max_clique) and clique not in visited_cliques:
                    subset_cliques.append(clique)
                    visited_cliques.add(clique)
                    cliques_assigned[tuple(max_clique)].append(clique)
            potential_values = self.clique_potentials[subset_cliques[0]]
            depends_on = set(subset_cliques[0])
            for clique in subset_cliques[1:]:
                potential_values, depends_on = multiply_potentials(
                    potential_values, self.clique_potentials[clique], depends_on, set(clique)
                )
            union_sub_cliques = set()
            for sub_clique in subset_cliques:
                union_sub_cliques = union_sub_cliques.union(sub_clique)
            left_nodes = set(max_clique).difference(union_sub_cliques)
            for node in left_nodes:
                potn = [1, 1]
                potential_values, depends_on = multiply_potentials(
                    potential_values, potn, depends_on, [node]
                )

            max_clique_potentials[tuple(max_clique)] = potential_values

        self.max_clique_potentials = max_clique_potentials
        # print("Maximal Clique Potentials:")
        # for clique, potentials in self.max_clique_potentials.items():
        #     print(f"{clique}: {potentials}")

    def get_z_value(self):
        """
        Compute the partition function (Z value) of the graphical model.

        What to do here:
        ----------------
        - Implement the message passing algorithm to compute the partition function (Z value).
        - The Z value is the normalization constant for the probability distribution.

        Refer to the problem statement for details on computing the partition function.
        """

        def create_empty_message_dict(junction_tree):
            message_dict = {}
            for node in junction_tree:
                message_dict[node] = {}
                for neighbor in junction_tree[node]:
                    message_dict[node][neighbor] = None
            return message_dict

        def multiply_messages(potential, message, node, mesg_depends_on):
            potential_new = deepcopy(potential)
            for val in itertools.product([0, 1], repeat=len(node)):
                assignmt = dict(zip(node, val))
                targ_idx = 0
                for t in mesg_depends_on:
                    targ_idx = (targ_idx << 1) + assignmt[t]

                node_idx = 0
                for n in node:
                    node_idx = (node_idx << 1) + assignmt[n]

                potential_new[node_idx] *= message[targ_idx]

            return potential_new

        def condense_message(potential, node, compl_to_sum_on, to_sum_on):
            new_potential = []
            for val in itertools.product([0, 1], repeat=len(compl_to_sum_on)):
                sum_val = 0
                for val1 in itertools.product(
                    [0, 1], repeat=len(node) - len(compl_to_sum_on)
                ):
                    assignmt = dict(zip(compl_to_sum_on, val))
                    assignmt.update(dict(zip(to_sum_on, val1)))
                    idx = 0
                    for n in node:
                        idx = (idx << 1) + assignmt[n]
                    sum_val += potential[idx]
                new_potential.append(sum_val)

            return new_potential

        def send_message(junc_tree, potentials, message_dict):
            def find_leaves(junc_tree):
                leaves = set()
                for node in junc_tree:
                    if len(junc_tree[node]) == 1:
                        leaves.add(node)
                return leaves

            dynamic_mem = find_leaves(junc_tree)
            while True:
                new_dynamic_mem = set()
                for node in dynamic_mem:
                    for target in junc_tree[node]:
                        potn = deepcopy(potentials[tuple(node)])
                        if message_dict[target][node] != None:
                            mes = message_dict[target][node]
                            potn = multiply_messages(potn, mes, node, set(node).intersection(target))

                    for target in junc_tree[node]:
                        if message_dict[target][node] == None:
                            diff = set(node).difference(target)
                            summing_nodes = set(node).difference(diff)
                            message_dict[node][target] = condense_message(
                                potn, node, summing_nodes, diff
                            )
                            new_dynamic_mem.add(target)

                if new_dynamic_mem == set():
                    print("Forward Pass Done")
                    break
                dynamic_mem = new_dynamic_mem

            return dynamic_mem

        def receive_message(junc_tree, potentials, message_dict, dynamic_mem):
            while True:
                new_dynamic_mem = set()
                for node in dynamic_mem:
                    for target in junc_tree[node]:
                        potn = deepcopy(potentials[tuple(node)])
                        if message_dict[node][target] == None:
                            for neigh in junc_tree[node]:
                                if neigh != target:
                                    potn = multiply_messages(potn, message_dict[neigh][node], node, set(node).intersection(neigh))
                            diff = set(node).difference(target)
                            summing_nodes = set(node).difference(diff)
                            message_dict[node][target] = condense_message(
                                potn, node, summing_nodes, diff
                            )
                            new_dynamic_mem.add(target)

                if new_dynamic_mem == set():
                    print("Backward Pass Done")
                    break
                dynamic_mem = new_dynamic_mem
        
        def calc_z(message_dict, potentials):
            node = list(potentials.keys())[0]
            z = potentials[node]
            for neigh in message_dict[node]:

                z = multiply_messages(z, message_dict[neigh][node], node, set(node).intersection(neigh))
            z = condense_message(z, node, [], node)
            return z[0]
        
        def brute_force_message_passing(junc_tree, potentials, max_cliques):
            message_dict = create_empty_message_dict(junc_tree)
            while(True):
                all_msgs = True
                for m_clique in message_dict:
                    for neigh in junc_tree[m_clique]:
                        if message_dict[m_clique][neigh] == None:
                            all_msgs = False
                            prereqs_met = True
                            for neigh1 in junc_tree[m_clique]:
                                if neigh1 != neigh and message_dict[neigh1][m_clique] == None:
                                    prereqs_met = False
                                    break
                            if not prereqs_met:
                                continue

                            potn = deepcopy(potentials[m_clique])
                            for neigh1 in junc_tree[m_clique]:
                                if neigh1 != neigh:
                                    potn = multiply_messages(potn, message_dict[neigh1][m_clique], m_clique, set(m_clique).intersection(neigh1))
                            diff = set(m_clique).difference(neigh)
                            summing_nodes = set(m_clique).difference(diff)
                            message_dict[m_clique][neigh] = condense_message(potn, m_clique, summing_nodes, diff)

                if all_msgs:
                    return message_dict
                
        def message_passing_opt_order(junc_tree, potentials, opt_order):
            message_dict = create_empty_message_dict(junc_tree)
            for m_clique, neigh in opt_order:
                potn = deepcopy(potentials[m_clique])
                for neigh1 in junc_tree[m_clique]:
                    if neigh1 != neigh:
                        potn = multiply_messages(potn, message_dict[neigh1][m_clique], m_clique, set(m_clique).intersection(neigh1))
                diff = set(m_clique).difference(neigh)
                summing_nodes = set(m_clique).difference(diff)
                message_dict[m_clique][neigh] = condense_message(potn, m_clique, summing_nodes, diff)
            return message_dict

        # def multiply_potentials(potential1, potential2, potn1_depends_on, potn2_depends_on):
        #             potential_new = [None] * (2 ** (len(set(potn1_depends_on).union(potn2_depends_on))))
        #             for val in itertools.product([0, 1], repeat=len(set(potn1_depends_on).union(potn2_depends_on))):
        #                 assignmt = dict(zip(set(potn1_depends_on).union(potn2_depends_on), val))
        #                 potn1_idx = 0
        #                 for t in potn1_depends_on:
        #                     potn1_idx = (potn1_idx << 1) + assignmt[t]

        #                 potn2_idx = 0
        #                 for t in potn2_depends_on:
        #                     potn2_idx = (potn2_idx << 1) + assignmt[t]

        #                 net_idx = 0
        #                 for t in set(potn1_depends_on).union(potn2_depends_on):
        #                     net_idx = (net_idx << 1) + assignmt[t]

        #                 potential_new[net_idx] = potential1[potn1_idx] * potential2[potn2_idx]

        #             return potential_new, set(potn1_depends_on).union(potn2_depends_on)

        # def brut_force():
        #     print(self.clique_potentials)
        #     potn = list(self.clique_potentials.values())[0]
        #     depends_on = set(list(self.clique_potentials.keys())[0])
        #     for clique in list(self.clique_potentials.keys())[1:]:
        #         print()
        #         print("multiplying", potn, self.clique_potentials[clique], depends_on, set(clique))
        #         potn, depends_on = multiply_potentials(potn, self.clique_potentials[clique], depends_on, set(clique))
        #     print("Brute Force Z:")
        #     print(potn, len(potn))
        #     z = condense_message(potn, depends_on, [], depends_on)
        #     print(z, sum(potn))
        #     print("-" * 50)
        #     return z[0]



        # brut_force()
        message_dict = message_passing_opt_order(self.junction_tree, self.max_clique_potentials, self.opt_order)
        # message_dict = brute_force_message_passing(self.junction_tree, self.max_clique_potentials, self.max_cliques)
        # message_dict = create_empty_message_dict(self.junction_tree)
        # print("Cliques and potentials:")
        # # print(self.clique_potentials)
        print("junction_tree:")
        print(self.junction_tree)
        # # print("Maximal Clique Potentials:")
        # # print(self.max_clique_potentials)
        # dynamic_mem = send_message(self.junction_tree, self.max_clique_potentials, message_dict)
        # receive_message(self.junction_tree, self.max_clique_potentials, message_dict, dynamic_mem)
        z = calc_z(message_dict, self.max_clique_potentials)
        print("message_dict:", message_dict)
        print("Z:", z)
        self.message_dict = message_dict
        self.z = z
        return z

    def compute_marginals(self):
        """
        Compute the marginal probabilities for all variables in the graphical model.

        What to do here:
        ----------------
        - Use the message passing algorithm to compute the marginal probabilities for each variable.
        - Return the marginals as a list of lists, where each inner list contains the probabilities for a variable.

        Refer to the sample test case for the expected format of the marginals.
        """

        def multiply_messages(potential, message, node, mesg_depends_on):
            potential_new = deepcopy(potential)
            for val in itertools.product([0, 1], repeat=len(node)):
                assignmt = dict(zip(node, val))
                targ_idx = 0
                for t in mesg_depends_on:
                    targ_idx = (targ_idx << 1) + assignmt[t]

                node_idx = 0
                for n in node:
                    node_idx = (node_idx << 1) + assignmt[n]

                potential_new[node_idx] *= message[targ_idx]

            return potential_new

        def condense_message(potential, node, compl_to_sum_on, to_sum_on):
            new_potential = []
            for val in itertools.product([0, 1], repeat=len(compl_to_sum_on)):
                sum_val = 0
                for val1 in itertools.product(
                    [0, 1], repeat=len(node) - len(compl_to_sum_on)
                ):
                    assignmt = dict(zip(compl_to_sum_on, val))
                    assignmt.update(dict(zip(to_sum_on, val1)))
                    idx = 0
                    for n in node:
                        idx = (idx << 1) + assignmt[n]
                    sum_val += potential[idx]
                new_potential.append(sum_val)

            return new_potential

        marginals = [None] * len(self.nodes)
        for i, node in enumerate(self.nodes):
            marginals[i] = [0, 0]
            super_clique = None
            for clique in self.max_cliques:
                if node in clique:
                    super_clique = clique
                    break

            neighbors = self.junction_tree[tuple(super_clique)]
            potential = deepcopy(self.max_clique_potentials[tuple(super_clique)])
            for neigh in neighbors:
                potential = multiply_messages(
                    potential, self.message_dict[neigh][tuple(super_clique)], super_clique, set(super_clique).intersection(neigh)
                )
            potential = condense_message(potential, super_clique, [node], set(super_clique).difference([node]))

            marginals[i][0] = potential[0] / self.z
            marginals[i][1] = potential[1] / self.z

        return marginals

    def compute_top_k(self):
        """
        Compute the top-k most probable assignments in the graphical model.

        What to do here:
        ----------------
        - Use the message passing algorithm to find the top-k assignments with the highest probabilities.
        - Return the assignments along with their probabilities in the specified format.

        Refer to the sample test case for the expected format of the top-k assignments.
        """

        def create_empty_message_dict(junction_tree):
            message_dict = {}
            for node in junction_tree:
                message_dict[node] = {}
                for neighbor in junction_tree[node]:
                    message_dict[node][neighbor] = None
            return message_dict
        
        def multiply_messages(potential, message, node, mesg_depends_on):
            potential_new = []
            for pot in potential:
                potential_new.append([])
            for val in itertools.product([0, 1], repeat=len(node)):
                assignmt = dict(zip(node, val))
                targ_idx = 0
                for t in mesg_depends_on:
                    targ_idx = (targ_idx << 1) + assignmt[t]

                node_idx = 0
                for n in node:
                    node_idx = (node_idx << 1) + assignmt[n]

                potential_assignment = potential[node_idx]
                message_assignment = message[targ_idx]
                set_potential_assignment = set(potential_assignment[0][1].keys())
                set_message_assignment = set(message_assignment[0][1].keys())

                intersection = set_potential_assignment.intersection(set_message_assignment)

                for pot_assignment in potential_assignment:
                    for mesg_assignment in message_assignment:
                        intersection_matches = True
                        for var in intersection:
                            if pot_assignment[1][var] != mesg_assignment[1][var]:
                                intersection_matches = False
                                break
                        if intersection_matches:
                            diff_dict = {var: mesg_assignment[1][var] for var in set_message_assignment.difference(intersection)}
                            new_assignment = deepcopy(pot_assignment)
                            new_assignment = list(new_assignment)
                            new_assignment[0] *= mesg_assignment[0]
                            new_assignment[1].update(diff_dict)
                            new_assignment = tuple(new_assignment)
                            potential_new[node_idx].append(new_assignment)
                
            return potential_new
        
        def condense_message(potential, node, compl_to_topk_on, to_topk_on):
            new_potential = []
            for val in itertools.product([0, 1], repeat=len(compl_to_topk_on)):
                topk_things = []
                for val1 in itertools.product([0, 1], repeat=len(to_topk_on)):
                    assignmt = dict(zip(compl_to_topk_on, val))
                    assignmt.update(dict(zip(to_topk_on, val1)))
                    idx = 0
                    for n in node:
                        idx = (idx << 1) + assignmt[n]

                    potential_assignment = potential[idx]
                    for pot_assgn in potential_assignment:
                        pot_assgn[1].update(dict(zip(to_topk_on, val1)))
                        topk_things.append(pot_assgn)
                topk_things.sort(key=lambda x: -x[0])
                topk_things = topk_things[:self.k]
                new_potential.append(topk_things)
            return new_potential

        
        def brute_force_message_passing_top_k(junc_tree, potentials, max_cliques):
            message_dict = create_empty_message_dict(junc_tree)
            while(True):
                all_msgs = True
                for m_clique in message_dict:
                    for neigh in junc_tree[m_clique]:
                        if message_dict[m_clique][neigh] == None:
                            all_msgs = False
                            prereqs_met = True
                            for neigh1 in junc_tree[m_clique]:
                                if neigh1 != neigh and message_dict[neigh1][m_clique] == None:
                                    prereqs_met = False
                                    break
                            if not prereqs_met:
                                continue

                            potn = deepcopy(potentials[m_clique])
                            potn_message = []
                            for pot in potn:
                                potn_message.append([(pot, {})])

                            for neigh1 in junc_tree[m_clique]:
                                if neigh1 != neigh:
                                    potn_message = multiply_messages(potn_message, message_dict[neigh1][m_clique], m_clique, set(m_clique).intersection(neigh1))
                            diff = set(m_clique).difference(neigh)
                            summing_nodes = set(m_clique).difference(diff)
                            message_dict[m_clique][neigh] = condense_message(potn_message, m_clique, summing_nodes, diff)

                if all_msgs:
                    return message_dict
                
        def message_passing_opt_order_top_k(junc_tree, potentials, opt_order):
            message_dict = create_empty_message_dict(junc_tree)
            for m_clique, neigh in opt_order:
                potn = deepcopy(potentials[m_clique])
                potn_message = []
                for pot in potn:
                    potn_message.append([(pot, {})])
                for neigh1 in junc_tree[m_clique]:
                    if neigh1 != neigh:
                        potn_message = multiply_messages(potn_message, message_dict[neigh1][m_clique], m_clique, set(m_clique).intersection(neigh1))
                diff = set(m_clique).difference(neigh)
                summing_nodes = set(m_clique).difference(diff)
                message_dict[m_clique][neigh] = condense_message(potn_message, m_clique, summing_nodes, diff)
            return message_dict
                
        # message_dict = brute_force_message_passing_top_k(self.junction_tree, self.max_clique_potentials, self.max_cliques)
        message_dict = message_passing_opt_order_top_k(self.junction_tree, self.max_clique_potentials, self.opt_order)
                
        sample_node = tuple(self.max_cliques[0])
        sample_node_potential = deepcopy(self.max_clique_potentials[sample_node])
        sample_node_potential = [[(pot, {})] for pot in sample_node_potential]
        for neigh in self.junction_tree[sample_node]:
            sample_node_potential = multiply_messages(sample_node_potential, message_dict[neigh][sample_node], sample_node, set(sample_node).intersection(neigh))

        top_k_assignments = condense_message(sample_node_potential, sample_node, [], sample_node)

        top_k_dict_list = []
        for assgn in top_k_assignments[0]:
            assign_list = []
            for key in sorted(assgn[1].keys()):
                assign_list.append(assgn[1][key])
            val = assgn[0]/self.z
            top_k_dict_list.append({
                "assignment": assign_list,
                "probability": val
            })

        return top_k_dict_list
            

                            


########################################################################

# Do not change anything below this line

########################################################################


class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, "r") as file:
            self.data = json.load(file)

    def get_output(self):
        n = len(self.data)
        output = []
        for i in range(n):
            print(f"Testcase {i+1}")
            inference = Inference(self.data[i]["Input"])
            inference.triangulate_and_get_cliques()
            inference.get_junction_tree()
            inference.assign_potentials_to_cliques()
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
            top_k_assignments = inference.compute_top_k()
            output.append(
                {
                    "Marginals": marginals,
                    "Top_k_assignments": top_k_assignments,
                    "Z_value": z_value,
                }
            )
            print("-" * 50)
        self.output = output

    def write_output(self, file_name):
        with open(file_name, "w") as file:
            json.dump(self.output, file, indent=4)


if __name__ == "__main__":
    # evaluator = Get_Input_and_Check_Output("Sample_Testcase.json")
    # evaluator = Get_Input_and_Check_Output("Test.json")
    evaluator = Get_Input_and_Check_Output("Testcases.json")
    evaluator.get_output()
    evaluator.write_output("Sample_Testcase_Output.json")
