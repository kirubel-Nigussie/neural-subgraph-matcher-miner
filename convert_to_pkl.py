import networkx as nx
import pickle
import os

def convert_txt_to_pkl(txt_path, pkl_path):
    print(f"Reading from {txt_path}...")
    
    # Read the edge list
    # CA-GrQc is an undirected collaboration network
    G = nx.read_edgelist(txt_path, nodetype=int, create_using=nx.Graph)
    
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Save as pickle
    print(f"Saving to {pkl_path}...")
    with open(pkl_path, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Conversion complete!")

if __name__ == "__main__":
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
        
    convert_txt_to_pkl('data/CA-GrQc.txt', 'data/CA-GrQc.pkl')
