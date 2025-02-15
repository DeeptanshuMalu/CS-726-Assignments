import json
from copy import deepcopy
import itertools

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

        for i in data["Cliques and Potentials"]:
            self.clique_potentials[tuple(i["cliques"])] = i["potentials"]
        for i in self.clique_potentials:
            for j in i:
                for k in i:
                    if j < k:
                        self.edges.add((j, k))
        for i in self.edges:
            for j in i:
                self.nodes.add(j)

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
                if adjlist[node] and is_simplicial(adjlist, node):
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

        self.junction_tree = make_junction_tree(self.junction_graph)
        # print("Junction Tree:")
        # for node, neighbors in self.junction_tree.items():
        #     print(f"{node}: {neighbors}")
        # print(make_junction_tree(p))

    def assign_potentials_to_cliques(self):
        """
        Assign potentials to the cliques in the junction tree.

        What to do here:
        ----------------
        - Map the given potentials (from the input data) to the corresponding cliques in the junction tree.
        - Ensure the potentials are correctly associated with the cliques for message passing.

        Refer to the sample test case for how potentials are associated with cliques.
        """
        max_clique_potentials = {}
        for max_clique in self.max_cliques:
            subset_clique_potentials = {}
            for clique in self.clique_potentials:
                if set(clique).issubset(max_clique):
                    subset_clique_potentials[clique] = self.clique_potentials[clique]

            # For each possible binary assignment to variables in max_clique
            max_clique_list = list(max_clique)
            potential_values = []

            # Generate all possible binary combinations for variables in max_clique
            for values in itertools.product([0, 1], repeat=len(max_clique_list)):
                assignment = dict(zip(max_clique_list, values))

                # Initialize potential for this assignment
                potential = 1

                # Multiply potentials from all subset cliques
                for clique, subset_clique_potential in subset_clique_potentials.items():
                    # Get the index in potentials for this assignment
                    idx = 0
                    bin_index = ""
                    for var in clique:
                        bin_index += str(assignment[var])
                    idx = int(bin_index, 2)
                    potential *= subset_clique_potential[idx]

                potential_values.append(potential)

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

        def multiply_messages(potential, message, node, target):
            potential_new = deepcopy(potential)
            for val in itertools.product([0, 1], repeat=len(node)):
                assignmt = dict(zip(node, val))
                targ_idx = 0
                for t in target:
                    targ_idx = (targ_idx << 1) + assignmt[t]

                node_idx = 0
                for n in node:
                    node_idx = (node_idx << 1) + assignmt[n]

                potential_new[node_idx] *= message[targ_idx]

            return potential_new

        def condense_message(potential, node, summing_nodes, diff):
            new_potential = []
            for val in itertools.product([0, 1], repeat=len(summing_nodes)):
                sum_val = 0
                for val1 in itertools.product(
                    [0, 1], repeat=len(node) - len(summing_nodes)
                ):
                    assignmt = dict(zip(summing_nodes, val))
                    assignmt.update(dict(zip(diff, val1)))
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
            while 1:
                new_dynamic_mem = set()
                for node in dynamic_mem:
                    potn = None
                    for target in junc_tree[node]:
                        if message_dict[target][node] != None:
                            mes = message_dict[target][node]
                            potn = potentials[tuple(node)]
                            potn = multiply_messages(potn, mes, node, set(node).intersection(target))

                    for target in junc_tree[node]:
                        if message_dict[target][node] == None:
                            diff = set(node).difference(target)
                            potn = potentials[tuple(node)]
                            summing_nodes = set(node).difference(diff)
                            message_dict[node][target] = condense_message(
                                potn, node, summing_nodes, diff
                            )
                            new_dynamic_mem.add(target)

                if new_dynamic_mem == set():
                    print("Forward Pass Done")
                    print(message_dict)
                    break
                dynamic_mem = new_dynamic_mem

            return dynamic_mem

        def receive_message(junc_tree, potentials, message_dict, dynamic_mem):
            while 1:
                new_dynamic_mem = set()
                for node in dynamic_mem:
                    for target in junc_tree[node]:
                        potn = potentials[tuple(node)]
                        if message_dict[target][node] != None and message_dict[node][target] == None:
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
                    print(message_dict)
                    break
        
        def calc_z(message_dict, potentials, max_cliques):
            node = list(max_cliques.keys())[0]
            z = potentials[node]
            for neigh in message_dict[node]:
                z = multiply_messages(z, message_dict[neigh][node], node, set(node).intersection(neigh))
            z = condense_message(z, node, [], node)
            return z[0]

        message_dict = create_empty_message_dict(self.junction_tree)
        print(self.junction_tree)
        print(self.max_clique_potentials)
        dynamic_mem = send_message(self.junction_tree, self.max_clique_potentials, message_dict)
        receive_message(self.junction_tree, self.max_clique_potentials, message_dict, dynamic_mem)
        z = calc_z(message_dict, self.max_clique_potentials, self.max_clique_potentials)
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
        pass

    def compute_top_k(self):
        """
        Compute the top-k most probable assignments in the graphical model.

        What to do here:
        ----------------
        - Use the message passing algorithm to find the top-k assignments with the highest probabilities.
        - Return the assignments along with their probabilities in the specified format.

        Refer to the sample test case for the expected format of the top-k assignments.
        """
        pass


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
    evaluator = Get_Input_and_Check_Output("Testcases.json")
    evaluator.get_output()
    evaluator.write_output("Sample_Testcase_Output.json")
