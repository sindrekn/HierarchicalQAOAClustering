The start of a ReadMe file: 

Ex run of the HierarchicalClusteringQAOA_ClassicalSimualtion.py file to run both a Hierarchical QAOA cluster solver and analyse the QAOA statevector: 
python HierarchicalClusteringQAOA_ClassicalSimualtion.py --hierarchical --test --n_restarts 100 --graphfile "graphs/XXSgraph.csv" --hierarchical_result "results/XXSgraph-Hierarchical.h5" --test_result "results/XXSgraph-analyse.h5"

Ex use og the file TreeStructurePlot:
python TreeStructurePlot.py --result_path "results/XXSgraph-Hierarchical.h5" --save_path "results/tree_plot_smallkarateclub.pdf"

Ex use of the file PlotClusters.py to plot a heatmap, probability histogram and the graph solution: 
python PlotClusters.py --heatmap --optimal-config --prob-dist --save_path "results/plot.pdf" --analyse_path results/XXSgraph-analyse.h5 --hierarchical_path results/XXSgraph-Hierarchical.h5 --graph_path "graphs/XXSgraph.csv"
