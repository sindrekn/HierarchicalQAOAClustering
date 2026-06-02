import networkx as nx
import matplotlib.pyplot as plt

# graph = nx.Graph()
# # edges = [(0, 1), (0, 2), (0, 6), (1, 2), (1, 6), (2, 3), (3, 4), (3, 5), (4, 5)]
# edges = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 4), (3, 8), (4, 5), (4, 6), (5, 6), (6, 7), (6, 9), (7, 8), (7, 9), (8, 9)]

# for edge in edges:
#     graph.add_edge(edge[0], edge[1])

# # graph = nx.karate_club_graph()

# G_karate = nx.karate_club_graph()
# # Take the core of community 0 + a few community 1 nodes
# nodes = list(range(12))  # or pick specific nodes
# G = G_karate.subgraph(nodes).copy()

# # Graph to a csv files of form Node1, Node2, Weight
# with open("graphs/HomeMadeGraph3.csv", "w") as f:
#     f.write("Node1,Node2,Weight\n")
#     for edge in G.edges():
#         f.write(f"{edge[0]},{edge[1]},1\n")

# Collect a graph from a csv file for plotting
import pandas as pd
graph = pd.read_csv("graphs/HomeMadeGraph3.csv")
edges   = [(row['Node1'], row['Node2']) for _, row in graph.iterrows()]
weights = [row['Weight'] for _, row in graph.iterrows()]

graph = nx.Graph()
for edge, weight in zip(edges, weights):
    graph.add_edge(edge[0], edge[1], weight=weight)

# Plot the graph
nx.draw(graph, with_labels=True)
plt.show()
