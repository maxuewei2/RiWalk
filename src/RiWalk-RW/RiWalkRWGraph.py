# -*- coding: utf-8 -*-
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from collections import defaultdict, deque


class RiGraph:
    def __init__(self, nx_g, args):
        self.g = nx_g
        self.num_walks = args.num_walks
        self.walk_length = args.walk_length
        self.workers = args.workers
        self.flag = args.flag
        self.discount = not args.without_discount
        logging.debug('discount option: {}'.format('On' if self.discount else 'Off'))

        self.num_nodes = len(self.g)
        self.degrees_ = tuple([len(self.g[_]) for _ in range(len(self.g))])
        self.logDegrees_ = tuple(np.asarray(self.degrees_).tolist())

    def bfs(self, rws, root, visited):
        """
        breadth-first search {self.g} from {root} following links in {rws}.
        mark each {node} visited with {visited[node]=root}.
        return dictionary of {node: its_distance_from_root} pairs.
        :param root: the anchor node.
        :param rws: random walks.
        :param visited: make sure nodes appeared in {rws}
                        being marked with {visited[node]=root}.
        :return: distance_dict
        """
        distance_dict = {root: 0}
        adj_dict = defaultdict(set)
        for walk in rws:
            for i, j in zip(walk[:-1], walk[1:]):
                adj_dict[i].add(j)
                adj_dict[j].add(i)
        last_layer = deque()
        last_layer.append(root)
        visited[root] = root
        layer_count = 1
        while True:
            next_layer = deque()
            for node_ in last_layer:
                neighbors = adj_dict[node_]
                for nb in neighbors:
                    if visited[nb] != root:
                        visited[nb] = root
                        next_layer.append(nb)
            if len(next_layer) == 0:
                return distance_dict
            for _ in next_layer:
                distance_dict[_] = layer_count
            last_layer = next_layer
            layer_count += 1

    def get_sp_dict(self, root, node_layer_dict, nb_nodes):
        """
        given a list of neighbor nodes {nb_nodes},
        return a dictionary of their new identifiers calculated by RiWalk-RWSP.
        :param root: the anchor node.
        :param node_layer_dict: a dictionary of {node: its_distance_from_root} pairs.
        :param nb_nodes: a list of neighbor nodes of root (with order).
        :return: sp_dict, a dictionary of {node: its_new_identifier} pairs
        """
        layer_list = [node_layer_dict[node] for node in nb_nodes]
        if self.discount:
            root_degree = self.logDegrees_[root]
            degree_list = [self.logDegrees_[node] for node in nb_nodes]
        else:
            root_degree = self.degrees_[root]
            degree_list = [self.degrees_[node] for node in nb_nodes]
        sp_dict = {node_: hash((root_degree, layer_, degree_)) for node_, layer_, degree_ in
                   zip(nb_nodes, layer_list, degree_list)}
        return sp_dict

    def get_wl_dict(self, root, node_layer_dict, nb_nodes, rws):
        """
        given a list of neighbor nodes {nb_nodes},
        return a dictionary of their new identifiers calculated by RiWalk-RWWL.
        :param root: the anchor node.
        :param node_layer_dict: a dictionary of {node: its_distance_from_root} pairs.
        :param nb_nodes: a list of neighbor nodes of root (with order).
        :param rws: random walks
        :return: wl_dict, a dictionary of {node: its_new_identifier} pairs
        """
        node_index_dict = {_: i for i, _ in enumerate(nb_nodes)}
        x_lists = [[0] * self.walk_length for _ in nb_nodes]
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
        wl_dict = {_: hash((root_x, node_layer_dict[_], tuple(x_lists[node_index_dict[_]]))) for _ in nb_nodes}
        return wl_dict

    def simulate_walk(self, walk_length, root):
        walk = [root]
        g = self.g
        rand = np.random.randint(self.num_nodes, size=walk_length)
        for i in range(walk_length-1):
            cur = walk[-1]
            cur_nbrs = g[cur]
            lc = len(cur_nbrs)
            if lc:
                next_node = cur_nbrs[rand[i] % lc]
                walk.append(next_node)
            else:
                break
        return walk

    def simulate_walks_for_node(self, root, num_walks, walk_length):
        walks = []
        for walk_iter in range(num_walks):
            walk = self.simulate_walk(walk_length=walk_length, root=root)
            walks.append(walk)
        return walks

    def process_random_walks(self):
        vertices = np.random.permutation(self.num_nodes).tolist()
        chunks = partition(list(vertices), self.workers)
        futures = {}
        total_walk_time_ = 0
        total_bfs_time_ = 0
        total_ri_time_ = 0
        total_walks_writing_time_ = 0
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            part = 1
            for c in chunks:
                job = executor.submit(process_random_walks_chunk, self, c, part,
                                      self.num_walks, self.walk_length)
                futures[job] = part
                part += 1
            for job in as_completed(futures):
                walk_time_, bfs_time_, ri_time_, walk_writing_time_ = job.result()
                total_walk_time_ += walk_time_
                total_bfs_time_ += bfs_time_
                total_ri_time_ += ri_time_
                total_walks_writing_time_ += walk_writing_time_
        return total_walk_time_, total_bfs_time_, total_ri_time_, total_walks_writing_time_


def get_ri_walks(walks, root, ri_dict):
    ri_walks = []
    ri_dict[root] = root
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


def process_random_walks_chunk(ri_graph, vertices, part_id, num_walks, walk_length):
    walks_all = []
    i = 0
    visited = [-1] * ri_graph.num_nodes
    walk_time_ = 0
    bfs_time_ = 0
    ri_time_ = 0
    walks_writing_time_ = 0
    for count, v in enumerate(vertices):
        walk_begin_time_ = time.time()
        walks = ri_graph.simulate_walks_for_node(v, num_walks, walk_length)
        walk_end_time_ = time.time()
        walk_time_ += (walk_end_time_ - walk_begin_time_)

        bfs_begin_time_ = time.time()
        node_layer_dict = ri_graph.bfs(walks, v, visited)
        bfs_end_time_ = time.time()
        bfs_time_ += (bfs_end_time_ - bfs_begin_time_)

        nei_nodes = list(node_layer_dict.keys())

        ri_begin_time_ = time.time()
        if 'sp' == ri_graph.flag:
            sp_dict = ri_graph.get_sp_dict(v, node_layer_dict, nei_nodes)
            sp_walks = get_ri_walks(walks, v, sp_dict)
            walks_all.extend(sp_walks)

        if 'wl' == ri_graph.flag:
            wl_dict = ri_graph.get_wl_dict(v, node_layer_dict, nei_nodes, walks)
            wl_walks = get_ri_walks(walks, v, wl_dict)
            walks_all.extend(wl_walks)
        ri_end_time_ = time.time()
        ri_time_ += (ri_end_time_ - ri_begin_time_)

        if count % 100 == 0:
            logging.debug('worker {} has processed {} nodes.'.format(part_id, count))

        walks_writing_begin_time_ = time.time()
        if len(walks_all) > 1000000:
            save_random_walks(walks_all, part_id, i)
            i += 1
            walks_all = []
        walks_writing_end_time_ = time.time()
        walks_writing_time_ += (walks_writing_end_time_ - walks_writing_begin_time_)
    walks_writing_begin_time_ = time.time()
    save_random_walks(walks_all, part_id, i)
    walks_writing_end_time_ = time.time()
    walks_writing_time_ += (walks_writing_end_time_ - walks_writing_begin_time_)
    return walk_time_, bfs_time_, ri_time_, walks_writing_time_


# https://github.com/leoribeiro/struc2vec
def partition(lst, n):
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]
