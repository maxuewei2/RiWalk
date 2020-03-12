# -*- coding: utf-8 -*-
"""
Reference implementation of RiWalk.
Author: Xuewei Ma
For more details, refer to the paper:
RiWalk: Fast Structural Node Embedding via Role Identification
ICDM, 2019
"""

import argparse
import FastRiGraph
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
import networkx as nx
import os
import glob
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

def parse_args():
    """
        Parses the RiWalk arguments.
    """
    parser = argparse.ArgumentParser(description="Run RiWalk")

    parser.add_argument('--input', nargs='?', default='graphs/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='embs/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num-walks', type=int, default=80,
                        help='Number of walks per source. Default is 80.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--until-k', type=int, default=4,
                        help='Neighborhood size k. Default is 4.')

    parser.add_argument('--iter', default=5, type=int,
                        help='Number of epochs in SGD. Default is 5.')

    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers. Default is 4.')

    parser.add_argument('--flag', nargs='?', default='sp',
                        help='Flag indicating using RiWalk-SP(sp) or RiWalk-WL(wl). Default is sp.')

    parser.add_argument('--discount', default=True, type=bool,
                        help='Flag indicating using discount.')

    return parser.parse_args()


class Sentences(object):
    def __init__(self, file_names):
        self.file_names = file_names

    def __iter__(self):
        fs = []
        for file_name in self.file_names:
            fs.append(open(file_name))
        while True:
            flag = 0
            for i, f in enumerate(fs):
                line = f.readline()
                if line != '':
                    flag = 1
                    yield line.split()
            if not flag:
                try:
                    for f in fs:
                        f.close()
                except:
                    pass
                return


class RiWalk:
    def __init__(self, args):
        self.args = args
        os.system('rm -rf walks/__random_walks_*.txt')

    def learn_embeddings(self):
        """
        learn embeddings from random walks.
        hs:  0:negative sampling 1:hierarchica softmax
        sg:  0:CBOW              1:skip-gram
        """
        dim = self.args.dimensions
        window_size = self.args.window_size
        workers = self.args.workers
        iter_num = self.args.iter

        walk_files = glob.glob('walks/__random_walks_*.txt')
        sentences = Sentences(walk_files)
        model = Word2Vec(sentences, size=dim, window=window_size, min_count=0, sg=1, workers=workers, iter=iter_num)
        return model.wv

    def read_graph(self):
        input_file_name = self.args.input
        nx_g = nx.read_edgelist(input_file_name, nodetype=int, create_using=nx.DiGraph())
        for edge in nx_g.edges():
            nx_g[edge[0]][edge[1]]['weight'] = 1
        nx_g = nx_g.to_undirected()
        return nx_g

    def preprocess_graph(self, nx_g):
        """
        1. relabel nodes with 0,1,2,3,...,N.
        2. convert graph to adjacency representation as a list of tuples.
        """
        mapping = {_: i for i, _ in enumerate(nx_g.nodes())}
        nx_g = nx.relabel_nodes(nx_g, mapping)
        nx_g = [tuple(nx_g.neighbors(_)) for _ in range(len(nx_g))]

        return nx_g, mapping

    def learn(self, nx_g, mapping):
        g = FastRiGraph.RiGraph(nx_g, self.args)

        g.process_random_walks()

        wv = self.learn_embeddings()

        original_wv = Word2VecKeyedVectors(self.args.dimensions)
        original_nodes = list(mapping.keys())
        original_vecs = [wv.word_vec(str(mapping[node])) for node in original_nodes]
        original_wv.add(entities=list(map(str, original_nodes)), weights=original_vecs)
        return original_wv

    def riwalk(self):
        nx_g = self.read_graph()
        nx_g, mapping = self.preprocess_graph(nx_g)
        return self.learn(nx_g, mapping)


def main():
    args = parse_args()
    print(vars(args))
    wv = RiWalk(args).riwalk()
    wv.save_word2vec_format(fname=args.output, binary=False)


if __name__ == '__main__':
    main()
