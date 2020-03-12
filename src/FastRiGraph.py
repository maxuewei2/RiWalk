# -*- coding: utf-8 -*-
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import logging

class RiGraph:
    def __init__(self, nx_g, args):
        self.g = nx_g
        self.num_walks = args.num_walks
        self.walk_length = args.walk_length
        self.workers = args.workers
        self.flag = args.flag
        self.discount = args.discount

        self.num_nodes = len(self.g)
        self.degrees_ = tuple([len(self.g[_]) for _ in range(len(self.g))])

        self.rand = random.Random()

    def get_dis(self, rws, start):
        dis = {start: 0}
        for walk in rws:
            for i, j in zip(walk[:-1], walk[1:]):
                disj = dis.get(j, 999999)
                disi = dis.get(i)
                if disj < disi:
                    dis[i] = disj + 1
                else:
                    dis[j] = min(disj, disi + 1)
        return dis

    def get_sp_dict(self, root, node_layer_dict, nei_nodes):
        layer_list = [node_layer_dict[node] for node in nei_nodes]
        degree_list = [self.degrees_[node] for node in nei_nodes]
        root_degree = self.degrees_[root]
        if self.discount:
            root_degree = simple_log2(root_degree + 1)
            degree_list = np.log2(np.asarray(degree_list) + 1).astype(np.int32).tolist()
        sp_dict = {node_: hash((root_degree, layer_, degree_)) for node_, layer_, degree_ in
                   zip(nei_nodes, layer_list, degree_list)}
        return sp_dict

    def get_wl_dict(self, root, node_layer_dict, nei_nodes, rws):
        node_index_dict = {_: i for i, _ in enumerate(nei_nodes)}
        x_lists = [[0] * self.walk_length for _ in nei_nodes]
        edge_set = set()
        for walk in rws:
            for i, j in zip(walk[:-1], walk[1:]):
                edge_set.add((i, j))
                edge_set.add((j, i))
        for i, j in edge_set:
            x_lists[node_index_dict[i]][node_layer_dict[j]] += 1
        if self.discount:
            x_lists = np.log2(np.asarray(x_lists) + 1).astype(np.int32).tolist()
        root_x = tuple(x_lists[node_index_dict[root]])
        wl_dict = {_: hash((root_x, node_layer_dict[_], tuple(x_lists[node_index_dict[_]]))) for _ in nei_nodes}
        return wl_dict

    def simulate_walk(self, walk_length, start_node, rand):
        walk = [start_node]
        g = self.g
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = g[cur]  # tuple(set(g[cur])&nei_nodes_set)
            if cur_nbrs:
                next_node = rand.choice(cur_nbrs)
                walk.append(next_node)
            else:
                break
        return walk

    def simulate_walks_for_node(self, node, num_walks, walk_length, rand):
        walks = []
        for walk_iter in range(num_walks):
            walk = self.simulate_walk(walk_length=walk_length, start_node=node, rand=rand)
            walks.append(walk)
        return walks

    def process_random_walks(self):
        num_walks, walk_length = self.num_walks, self.walk_length
        vertices = np.random.permutation(self.num_nodes).tolist()
        parts = self.workers
        chunks = partition(list(vertices), parts)
        futures = {}
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            part = 1
            for c in chunks:
                job = executor.submit(process_random_walks_chunk, self, c, part,
                                      num_walks, walk_length)
                futures[job] = part
                part += 1
            for job in as_completed(futures):
                job.result()
        return


def get_ri_walks(walks, start_node, ri_dict):
    ri_walks = []
    ri_dict[start_node] = start_node
    for walk in walks:
        ri_walk = [ri_dict[x] for x in walk]
        ri_walks.append(ri_walk)
    return ri_walks


def simple_log2(x):
    return x.bit_length() - 1


def save_random_walks(walks, part, i):
    indexes = np.random.permutation(len(walks)).tolist()
    with open('walks/__random_walks_{}_{}.txt'.format(part, i), 'w') as f:
        for i in indexes:
            walk = walks[i]
            f.write(u"{}\n".format(u" ".join(str(v) for v in walk)))


def process_random_walks_chunk(rigraph, vertices, part_id, num_walks, walk_length):
    walks_all = []
    i = 0
    rand = rigraph.rand
    for count, v in enumerate(vertices):
        walks = rigraph.simulate_walks_for_node(v, num_walks, walk_length, rand)
        node_layer_dict = rigraph.get_dis(walks, v)
        nei_nodes = list(node_layer_dict.keys())

        if 'sp' == rigraph.flag:
            sp_dict = rigraph.get_sp_dict(v, node_layer_dict, nei_nodes)
            sp_walks = get_ri_walks(walks, v, sp_dict)
            walks_all.extend(sp_walks)

        if 'wl' == rigraph.flag:
            wl_dict = rigraph.get_wl_dict(v, node_layer_dict, nei_nodes, walks)
            wl_walks = get_ri_walks(walks, v, wl_dict)
            walks_all.extend(wl_walks)
        
        if count%10==0:
            logging.debug('worker {} has processed {} nodes.'.format(part_id, count))
        if len(walks_all) > 100000:
            save_random_walks(walks_all, part_id, i)
            i += 1
            walks_all = []
    save_random_walks(walks_all, part_id, i)


# https://github.com/leoribeiro/struc2vec
def partition(lst, n):
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]
