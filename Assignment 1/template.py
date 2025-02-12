import json
from copy import deepcopy


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

        self.num_nodes = data["VariablesCount"]
        self.num_potentials = data["Potentials_count"]
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
        print("\nChordal Adjacency List:")
        for node, neighbors in self.adjlist.items():
            print(f"{node}: {neighbors}")
        print("Maximal Cliques:", self.max_cliques)

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
        pass

    def assign_potentials_to_cliques(self):
        """
        Assign potentials to the cliques in the junction tree.

        What to do here:
        ----------------
        - Map the given potentials (from the input data) to the corresponding cliques in the junction tree.
        - Ensure the potentials are correctly associated with the cliques for message passing.

        Refer to the sample test case for how potentials are associated with cliques.
        """
        pass

    def get_z_value(self):
        """
        Compute the partition function (Z value) of the graphical model.

        What to do here:
        ----------------
        - Implement the message passing algorithm to compute the partition function (Z value).
        - The Z value is the normalization constant for the probability distribution.

        Refer to the problem statement for details on computing the partition function.
        """
        pass

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
        self.output = output

    def write_output(self, file_name):
        with open(file_name, "w") as file:
            json.dump(self.output, file, indent=4)


if __name__ == "__main__":
    evaluator = Get_Input_and_Check_Output("Sample_Testcase.json")
    evaluator.get_output()
    evaluator.write_output("Sample_Testcase_Output.json")
