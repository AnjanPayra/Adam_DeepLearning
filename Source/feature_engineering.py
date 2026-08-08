"""
Step 1: Feature Engineering
Builds topological + complex-membership features for every protein in the
YDIP protein-protein interaction (PPI) network, and merges them with the
essential / non-essential ground-truth labels.
"""
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load the PPI network (YDIP.csv -> edge list, no header)
# ---------------------------------------------------------------
edges = pd.read_csv('/mnt/user-data/uploads/YDIP.csv', header=None, names=['p1', 'p2'])
G = nx.from_pandas_edgelist(edges, 'p1', 'p2')
G.remove_edges_from(nx.selfloop_edges(G))
print(f"PPI network: {G.number_of_nodes()} proteins, {G.number_of_edges()} interactions")

# ---------------------------------------------------------------
# 2. Load the protein-complex membership network (complex_network.xlsx)
#    Each row = one complex, each cell = a member protein (ragged / NaN padded)
# ---------------------------------------------------------------
cx = pd.read_excel('/mnt/user-data/uploads/complex_network.xlsx', sheet_name='Sheet1', header=None)
complexes = []
for _, row in cx.iterrows():
    members = [str(x).strip() for x in row.tolist() if pd.notna(x)]
    if members:
        complexes.append(members)

prot2complex = defaultdict(list)
for i, c in enumerate(complexes):
    for p in c:
        prot2complex[p].append(i)

complex_sizes = np.array([len(c) for c in complexes])
print(f"Complex network: {len(complexes)} complexes, "
      f"{len(prot2complex)} proteins with complex annotations")

# Build a "co-complex" graph: two proteins linked if they share >=1 complex
CoG = nx.Graph()
CoG.add_nodes_from(G.nodes())
for c in complexes:
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            CoG.add_edge(c[i], c[j])

# ---------------------------------------------------------------
# 3. Topological features from the PPI graph
# ---------------------------------------------------------------
print("Computing PPI topological features ...")
degree      = dict(G.degree())
clustering  = nx.clustering(G)
betweenness = nx.betweenness_centrality(G, k=500, seed=RANDOM_STATE)  # approx, for speed
closeness   = nx.closeness_centrality(G)
pagerank    = nx.pagerank(G, alpha=0.85)
eigen       = nx.eigenvector_centrality(G, max_iter=1000)
avg_nbr_deg = nx.average_neighbor_degree(G)
core_number = nx.core_number(G)

# ---------------------------------------------------------------
# 4. Complex-membership features
# ---------------------------------------------------------------
def n_complexes(p):
    return len(prot2complex.get(p, []))

def mean_complex_size(p):
    idxs = prot2complex.get(p, [])
    return float(np.mean([len(complexes[i]) for i in idxs])) if idxs else 0.0

def max_complex_size(p):
    idxs = prot2complex.get(p, [])
    return float(np.max([len(complexes[i]) for i in idxs])) if idxs else 0.0

co_degree      = dict(CoG.degree())          # # of distinct co-complex partners
co_clustering  = nx.clustering(CoG)

# ---------------------------------------------------------------
# 5. Assemble feature table
# ---------------------------------------------------------------
rows = []
for p in G.nodes():
    rows.append({
        'protein': p,
        'degree': degree.get(p, 0),
        'clustering_coeff': clustering.get(p, 0.0),
        'betweenness': betweenness.get(p, 0.0),
        'closeness': closeness.get(p, 0.0),
        'pagerank': pagerank.get(p, 0.0),
        'eigenvector_centrality': eigen.get(p, 0.0),
        'avg_neighbor_degree': avg_nbr_deg.get(p, 0.0),
        'k_core': core_number.get(p, 0),
        'n_complexes': n_complexes(p),
        'mean_complex_size': mean_complex_size(p),
        'max_complex_size': max_complex_size(p),
        'co_complex_degree': co_degree.get(p, 0),
        'co_complex_clustering': co_clustering.get(p, 0.0),
    })
feat_df = pd.DataFrame(rows)

# ---------------------------------------------------------------
# 6. Load essential / non-essential ground-truth labels
# ---------------------------------------------------------------
lab_path = '/home/claude/work/Essential_and_non_essential_dataset.xlsx'
ess     = pd.read_excel(lab_path, sheet_name='essential proteins', header=None)[0].astype(str).str.strip()
noness  = pd.read_excel(lab_path, sheet_name='non-essential proteins', header=None)[0].astype(str).str.strip()

label_map = {p: 1 for p in ess}
label_map.update({p: 0 for p in noness})

feat_df['label'] = feat_df['protein'].map(label_map)

# Keep only proteins that have a ground-truth label
data = feat_df.dropna(subset=['label']).reset_index(drop=True)
data['label'] = data['label'].astype(int)

print(f"\nFinal labeled dataset: {data.shape[0]} proteins x {data.shape[1]-2} features")
print(data['label'].value_counts().rename({1: 'essential', 0: 'non-essential'}))

data.to_csv('/home/claude/work/protein_features.csv', index=False)
print("\nSaved -> /home/claude/work/protein_features.csv")
print(data.head())
