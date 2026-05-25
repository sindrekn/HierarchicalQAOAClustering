import networkx as nx

graph = nx.Graph()
# edges = [(0, 1), (0, 2), (0, 6), (1, 2), (1, 6), (2, 3), (3, 4), (3, 5), (4, 5)]
edges = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 4), (3, 8), (4, 5), (4, 6), (5, 6), (6, 7), (6, 9), (7, 8), (7, 9), (8, 9)]

for edge in edges:
    graph.add_edge(edge[0], edge[1])

# graph = nx.karate_club_graph()

# Graph to a csv files of form Node1, Node2, Weight
with open("result/graph.csv", "w") as f:
    f.write("Node1,Node2,Weight\n")
    for edge in graph.edges():
        f.write(f"{edge[0]},{edge[1]},1\n")
