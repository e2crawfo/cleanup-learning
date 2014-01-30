import networkx as nx
import matplotlib.pyplot as plt
from mytools import hrr
import random

def semantic_network(n, seed=1):
    #G = nx.scale_free_graph(n, alpha=0.05, gamma=0.41, delta_out=0.2, delta_in=0, seed=seed)
    G = nx.scale_free_graph(n,seed=seed)
    G.remove_edges_from(G.selfloop_edges())
    G = nx.DiGraph(G)

    print G
    print "Max out: ", max(G.out_degree(G.nodes_iter()).values())
    print G.out_degree(G.nodes_iter())
    print "Max in: ", max(G.in_degree(G.nodes_iter()).values())
    print G.in_degree(G.nodes_iter())

    return G

def draw_semantic_network(G, edge_labels=None):
    #pos = nx.spring_layout(G)
    pos = nx.circular_layout(G)
    nx.draw(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

def make_id_vectors(G, D):
    vocab = hrr.Vocabulary(D)
    id_vectors = {n: vocab.parse('node'+str(n)) for n in G.nodes_iter()}
    return id_vectors

def make_edge_vectors(G, D):
    vocab = hrr.Vocabulary(D)
    max_out_degree = max(G.out_degree(G.nodes_iter()).values())
    edge_vectors = {i: vocab.parse('edge'+str(i)) for i in range(max_out_degree)}
    return edge_vectors

def make_hrr_vectors(G, id_vectors, edge_vectors):
    max_out_degree = len(edge_vectors)

    index_dict = {}
    for n in G.nodes_iter():
        edges = G.edges(n)
        indices = random.sample(range(max_out_degree), len(edges))
        index_dict.update(dict(zip(edges, indices)))

    nx.set_edge_attributes(G, 'index', index_dict)

    hrrvecs = {}
    for n in G.nodes_iter():
        edges = G.edges(n)
        if not edges:
            hrr_vec = hrr.HRR(len(id_vectors.values()[0]))
        else:
            components = [id_vectors[e[1]].convolve(edge_vectors[index_dict[e]])
                          for e
                          in edges]
            hrrvec = sum(components[1:], components[0])
            hrrvec.normalize()

        hrrvecs[n] = hrrvec

    return hrrvecs

def set_edge_weights(G, hrr_vectors, id_vectors, edge_vectors):

    index_dict = nx.get_edge_attributes(G, 'index')
    weight_dict = {}

    for n in G.nodes_iter():
        edges = G.edges(n)

        target_vectors = [id_vectors[e[1]] for e in edges]

        hrr_vec = hrr_vectors[n]
        trace_vectors = [hrr_vec.convolve(~edge_vectors[index_dict[e]]) for e in edges]

        weights = [round(target.compare(trace), 4)
                   for target, trace
                   in zip(target_vectors, trace_vectors)]
        weight_dict.update(dict(zip(edges, weights)))

    nx.set_edge_attributes(G, 'weight', weight_dict)
    return weight_dict

def build_semantic_network(D, N, seed=1, draw=False):
    random.seed(seed)
    G = semantic_network(N, seed=seed)

    id_vectors = make_id_vectors(G, D)

    edge_vectors = make_edge_vectors(G, D)

    hrr_vectors = make_hrr_vectors(G, id_vectors, edge_vectors)
    weight_dict = set_edge_weights(G, hrr_vectors, id_vectors, edge_vectors)

    if draw:
        draw_semantic_network(G, edge_labels=weight_dict)
        plt.show()

    return hrr_vectors, id_vectors, edge_vectors, G

if __name__ == '__main__':
    seed = 100
    D = 512
    N = 30

    build_semantic_network(D, N, seed, draw=True)

