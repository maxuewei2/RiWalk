/*
A reference implementation of RiWalk.
Author: Xuewei Ma
For more details, refer to the paper:
  RiWalk: Fast Structural Node Embedding via Role Identification, ICDM, 2019.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <stdatomic.h>

#define MAX_STRING 500
#define MAX_NUM_WALKS 10000000
#define LOG2(X) ((unsigned) (8*sizeof (unsigned long long) - __builtin_clzll((X)) - 1))

typedef float real;                    // Precision of float numbers
typedef long long bigint;
typedef struct node {
    char *name;
    int *ngbrs;
    int degree;
} node;

const int hash_table_size = 30000000;

char network_file[MAX_STRING], embedding_file[MAX_STRING], walk_dir[MAX_STRING], flag[20];
int num_threads = 4, dim = 128, until_k = 4, num_walks = 80, walk_length = 10, window_size = 10, iter_num = 5, discount = 1;
int *node_name_hash_table;
node *nodes;
int max_num_nodes = 1000, num_nodes;
bigint num_edges;
real *bfs_times, *ri_times, *walk_times;
atomic_int riwalk_i;

/* Build a hash table, mapping each node name to a unique node id */
unsigned int Hash(char *key, int n) {
    unsigned int seed = 131;
    unsigned int hash = 0;
    if (n == 0) {
        while (*key) {
            hash = hash * seed + (*key++);
        }
    } else {
        for (int i = 0; i < n; i++) {
            hash = hash * seed + (*key++);
        }
    }
    return hash % hash_table_size;
}

void InitHashTable() {
    node_name_hash_table = (int *) malloc(hash_table_size * sizeof(int));
    for (int k = 0; k != hash_table_size; k++) node_name_hash_table[k] = -1;
}

void InsertHashTable(char *key, int value) {
    unsigned int addr = Hash(key, 0);
    while (node_name_hash_table[addr] != -1) addr = (addr + 1) % hash_table_size;
    node_name_hash_table[addr] = value;
}

int SearchHashTable(char *key) {
    unsigned int addr = Hash(key, 0);
    while (1) {
        if (node_name_hash_table[addr] == -1) return -1;
        if (!strcmp(key, nodes[node_name_hash_table[addr]].name)) return node_name_hash_table[addr];
        addr = (addr + 1) % hash_table_size;
    }
}

int AddNode(char *name) {
    unsigned long length = strlen(name) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    nodes[num_nodes].name = (char *) calloc(length, sizeof(char));
    strncpy(nodes[num_nodes].name, name, length - 1);
    nodes[num_nodes].degree = 0;
    num_nodes++;
    if (num_nodes + 2 >= max_num_nodes) {
        max_num_nodes = (int) (max_num_nodes * 1.5);
        node *nodes_tmp = (node *) realloc(nodes, max_num_nodes * sizeof(node));
        if (nodes_tmp == NULL) {
            perror("Error: memory allocation failed!\n");
            exit(1);
        }
        nodes = nodes_tmp;
    }
    InsertHashTable(name, num_nodes - 1);
    return num_nodes - 1;
}

/* Read network from the training file
 * Give each node a new id, counting from 0 to {num_nodes}
 * */
void ReadData() {
    FILE *fin;
    char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2 * MAX_STRING + 10000];
    int vid;

    fin = fopen(network_file, "rb");
    if (fin == NULL) {
        perror("ERROR: network file not found!\n");
        exit(1);
    }
    num_edges = 0;
    while (fgets(str, sizeof(str), fin)) num_edges++;
    fclose(fin);
    printf("Number of edges: %lld          \n", num_edges);

    int *edge_source_id = (int *) malloc(num_edges * sizeof(int));
    int *edge_target_id = (int *) malloc(num_edges * sizeof(int));
    if (edge_source_id == NULL || edge_target_id == NULL) {
        perror("Error: memory allocation failed!\n");
        exit(1);
    }
    fin = fopen(network_file, "r");
    num_nodes = 0;
    for (bigint k = 0; k != num_edges; k++) {
        fscanf(fin, "%s %s", name_v1, name_v2);

        if (k % 10000 == 0) {
            printf("Reading edges: %.3lf%%%c", k / (double) (num_edges + 1) * 100, 13);
            fflush(stdout);
        }

        vid = SearchHashTable(name_v1);
        if (vid == -1) vid = AddNode(name_v1);
        nodes[vid].degree += 1;
        edge_source_id[k] = vid;

        vid = SearchHashTable(name_v2);
        if (vid == -1) vid = AddNode(name_v2);
        nodes[vid].degree += 1;
        edge_target_id[k] = vid;
    }
    fclose(fin);
    printf("Number of nodes: %d          \n", num_nodes);

    int *nodes_ngbr_count = (int *) calloc(num_nodes, sizeof(int));
    for (int i = 0; i < num_nodes; i++) {
        nodes[i].ngbrs = (int *) malloc(nodes[i].degree * sizeof(int));
    }
    for (bigint i = 0; i < num_edges; i++) {
        int node1 = edge_source_id[i], node2 = edge_target_id[i];
        nodes[node1].ngbrs[nodes_ngbr_count[node1]] = node2;
        nodes_ngbr_count[node1]++;
        nodes[node2].ngbrs[nodes_ngbr_count[node2]] = node1;
        nodes_ngbr_count[node2]++;
    }
    free(edge_source_id);
    free(edge_target_id);
    free(nodes_ngbr_count);
}

/* Simulate a random walk from {root}.
 * Save the walk in {walks}.
 * Save the length of the walk in {walk_len}*/
int SimulateWalk(node *g, int root, const int *tmp_list, int *walks, int *walk_len, int n) {
    walks += (n * walk_length);
    walks[0] = root;
    int current = root, i;
    for (i = 1; i < walk_length; i++) {
        int tl = tmp_list[current];
        if (tl != 0) {
            current = g[current].ngbrs[rand() % tl];
            walks[i] = current;
        } else {
            break;
        }
    }
    walk_len[n] = i;
    return n + 1;
}

int SimulateWalksForNode(node *g, int root, int *tmp_list, int *walks, int *walk_len, int n) {
    for (int i = 0; i < num_walks; i++) {
        n = SimulateWalk(g, root, tmp_list, walks, walk_len, n);
    }
    return n;
}

void swap(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

/* Given the anchor node {v}, the context node {nbj},
 * make the first {tmp_list[nbj]} nodes of {g[nbj].ngbrs} are within {until_k} hops from {v}
 * */
void ProcessNodeInTheLastLayer(node *g, int v, int nbj, const int *visited, int *tmp_list) {
    int p = 0;
    int ln = g[nbj].degree - 1;
    int q = ln;
    int *ngbrs = g[nbj].ngbrs;
    while (p <= q) {
        while (q >= 0 && visited[ngbrs[q]] != v)
            q -= 1;
        while (p <= ln && visited[ngbrs[p]] == v)
            p += 1;
        if (p < q)
            swap(ngbrs + p, ngbrs + q);
    }
    tmp_list[nbj] = p;
}

/* breadth-first search graph {g} from {v} and mark each context node {nbl} with {visited[nbl]=v}.
 * save the context nodes in {ngbrs[:nbgrCount]} and their distance from {v} in {distance}.*/
int BFS(node *g, int v, int *visited, int *tmp_list, int *ngbrs, int *distance) {
    visited[v] = v;
    ngbrs[0] = v;
    distance[v] = 0;
    int toVisitPtr = 0;
    int ngbrCount = 1;
    while (toVisitPtr < ngbrCount) {
        int nbj = ngbrs[toVisitPtr];
        if (distance[nbj] == until_k) {
            ProcessNodeInTheLastLayer(g, v, nbj, visited, tmp_list);
        } else {
            for (int i = 0; i < g[nbj].degree; i++) {
                int nbl = g[nbj].ngbrs[i];
                if (visited[nbl] != v) {
                    visited[nbl] = v;
                    ngbrs[ngbrCount] = nbl;
                    distance[nbl] = distance[nbj] + 1;
                    ngbrCount++;
                }
            }
            tmp_list[nbj] = g[nbj].degree;
        }
        toVisitPtr++;
    }
    return ngbrCount;
}

/* calculate new identifier for node {nb}*/
int GetSp(node *g, int nb, int hash_n, int *ri_arr, const int *distance) {
    int dis = distance[nb];
    int dgr = g[nb].degree;
    if (discount) {
        dgr = LOG2(dgr + 1);
    }
    ri_arr[1] = dis;
    ri_arr[2] = dgr;
    return (int) -Hash((char *) ri_arr, hash_n);
}

/* replace nodes in random walks with their new identifiers*/
void SpWalk(node *g, int v, int *walks, const int *walk_len, int n_bak, int n, const int *distance, int *tmp_list) {
    tmp_list[v] = v;
    int root_dgr = g[v].degree, len, tmp, nb;
    if (discount) { root_dgr = LOG2(root_dgr + 1); }
    int ri_arr[4] = {root_dgr, 0, 0, 0};
    int hash_n = sizeof(int) / sizeof(char) * 3;
    for (int i = n_bak; i < n; i++) {
        len = walk_len[i];
        for (int j = 0; j < len; j++) {
            tmp = i * walk_length + j;
            nb = walks[tmp];
            if (nb != v && tmp_list[nb] >= 0) {
                tmp_list[nb] = GetSp(g, nb, hash_n, ri_arr, distance);
            }
            walks[tmp] = tmp_list[nb];
        }
    }
}

void WriteWalks(int part, int write_count, const int *walks, int n, const int *walk_len) {
    if(n==0) {
        return ;
    }
    // shuffle random walks
    int *permutation = (int *) malloc(n * sizeof(int));
    //memset(permutation, -1, n * sizeof(int));
    int x, i, j;
    for (i = 0; i < n; i++) {
        permutation[i] = i;
    }
    for (i = 0; i < n; i++) {
        swap(permutation + i, permutation + (rand() % n));
    }

    char tmp[MAX_STRING];
    sprintf(tmp, "%s/__random_walks_%d_%d__.txt", walk_dir, part, write_count);
    FILE *fout = fopen(tmp, "w");
    if (fout == NULL) {
        perror("ERROR: can't open random walk file to write!\n");
        exit(1);
    }
    for (i = 0; i < n; i++) {
        x = permutation[i];
        for (j = 0; j < walk_len[x]; j++) {
            fprintf(fout, "%d ", walks[x * walk_length + j]);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);
    free(permutation);
}

/* deepcopy a graph. but the names of nodes are not deepcopied*/
node *CopyGraphWithoutName() {
    node *g = (node *) malloc(num_nodes * sizeof(node));
    if (g == NULL) {
        perror("memory allocation error.\n");
        exit(-1);
    }
    //memset(g, 0, num_nodes * sizeof(node));
    memcpy(g, nodes, num_nodes * sizeof(node));
    for (int i = 0; i < num_nodes; i++) {
        g[i].ngbrs = (int *) malloc(nodes[i].degree * sizeof(int));
        if (g[i].ngbrs == NULL) {
            perror("memory allocation error.\n");
            exit(-1);
        }
        //memset(g[i].ngbrs, 0, nodes[i].degree * sizeof(int));
        memcpy(g[i].ngbrs, nodes[i].ngbrs, nodes[i].degree * sizeof(int));
    }
    return g;
}

void FreeGraph(node *g) {
    for (int i = 0; i < num_nodes; i++) {
        free(g[i].ngbrs);
    }
    free(g);
}

void FreeThread(node *g, int *visited, int *tmp_list, int *ngbrs, int *distance, int *walks, int *walk_len) {
    free(tmp_list);
    free(visited);
    free(ngbrs);
    free(distance);
    free(walks);
    free(walk_len);
    FreeGraph(g);
}

void RiWalkThread(int part) {
    node *g = CopyGraphWithoutName();
    int *tmp_list = (int *) malloc(num_nodes*sizeof(int));
    int *visited = (int *) malloc(num_nodes * sizeof(int));
    int *ngbrs = (int *) malloc(num_nodes* sizeof(int));
    int *distance = (int *) malloc(num_nodes* sizeof(int));
    int *walks = (int *) malloc(MAX_NUM_WALKS * walk_length* sizeof(int));
    int *walk_len = (int *) malloc(MAX_NUM_WALKS* sizeof(int));
    if (visited == NULL || tmp_list == NULL || ngbrs == NULL || distance == NULL || walks == NULL || walk_len == NULL) {
        perror("memory allocation error.\n");
        exit(-1);
    }
    memset(visited, -1, num_nodes * sizeof(int));
    memset(tmp_list, 0, num_nodes * sizeof(int));
    memset(ngbrs, -1, num_nodes * sizeof(int));
    memset(distance, -1, num_nodes * sizeof(int));
    memset(walks, -1, MAX_NUM_WALKS * walk_length * sizeof(int));
    memset(walk_len, -1, MAX_NUM_WALKS * sizeof(int));
    int n = 0, write_count = 0;
    real bfs_time_ = 0, walk_time_ = 0, ri_time_ = 0, secs;
    struct timeval start, stop;
    int node_count = 0, v;
    while (riwalk_i < num_nodes) {
        v = atomic_fetch_add_explicit(&riwalk_i, 1, memory_order_relaxed);
        if (v > num_nodes)break;
        int n_bak = n;
        gettimeofday(&start, NULL);
        BFS(g, v, visited, tmp_list, ngbrs, distance);
        gettimeofday(&stop, NULL);
        secs = (double) (stop.tv_usec - start.tv_usec) / 1000000 + (double) (stop.tv_sec - start.tv_sec);
        bfs_time_ += secs;

        gettimeofday(&start, NULL);
        n = SimulateWalksForNode(g, v, tmp_list, walks, walk_len, n);
        gettimeofday(&stop, NULL);
        secs = (double) (stop.tv_usec - start.tv_usec) / 1000000 + (double) (stop.tv_sec - start.tv_sec);
        walk_time_ += secs;

        gettimeofday(&start, NULL);
        if (strcmp(flag, "sp") == 0 || strcmp(flag, "SP") == 0 || strcmp(flag, "Sp") == 0) {
            SpWalk(g, v, walks, walk_len, n_bak, n, distance, tmp_list);
        }
        gettimeofday(&stop, NULL);
        secs = (double) (stop.tv_usec - start.tv_usec) / 1000000 + (double) (stop.tv_sec - start.tv_sec);
        ri_time_ += secs;

        if (n + num_walks >= MAX_NUM_WALKS) {
            WriteWalks(part, write_count, walks, n, walk_len);
            n = 0;
            write_count++;
        }
        if (v % 1000 == 10) {
            printf("RiWalk process %.3lf%%%c", v * 100 / (double) num_nodes, 13);
            fflush(stdout);
        }
        node_count++;
    }
    WriteWalks(part, write_count, walks, n, walk_len);
    FreeThread(g, visited, tmp_list, ngbrs, distance, walks, walk_len);
    printf("part %d processed %d nodes in total\n", part, node_count);
    printf("part %d bfs time %f\n", part, bfs_time_);
    printf("part %d ri time %f\n", part, ri_time_);
    printf("part %d walk time %f\n", part, walk_time_);
    ri_times[part] = ri_time_;
    bfs_times[part] = bfs_time_;
    walk_times[part] = walk_time_;
    pthread_exit(NULL);
}

void RiWalk() {
    pthread_t *pt = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
    printf("--------------------------------\n");
    for (long a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, (void *(*)(void *)) RiWalkThread, (void *) a);
    for (long a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    free(pt);
}

void SaveEmbeddings() {
    char embedding_tmp_filename[MAX_STRING];
    sprintf(embedding_tmp_filename, "%s.tmp", embedding_file);
    FILE *tmp_emb = fopen(embedding_tmp_filename, "r");
    if (tmp_emb == NULL) {
        perror("ERROR: temp embedding file not found!\n");
        exit(1);
    }
    FILE *emb_file = fopen(embedding_file, "w");
    if (emb_file == NULL) {
        perror("ERROR: can not open embedding file to write!\n");
        exit(1);
    }
    int num_embs, node, num = 0, dim_t;
    real tmp;
    char tmp_str[50 * dim];
    fscanf(tmp_emb, "%d", &num_embs);
    fscanf(tmp_emb, "%d", &dim_t);
    if (dim_t != dim) {
        perror("ERROR: the dimensions of learned embeddings are not right.");
        exit(-1);
    }
    fprintf(emb_file, "%d %d\n", num_nodes, dim);
    for (int i = 0; i < num_embs; i++) {
        fscanf(tmp_emb, "%d", &node);
        if (node < 0) {
            fgets(tmp_str, sizeof(tmp_str), tmp_emb);
        } else {
            num++;
            fprintf(emb_file, "%s ", nodes[node].name);
            for (int j = 0; j < dim; j++) {
                fscanf(tmp_emb, "%f", &tmp);
                fprintf(emb_file, "%f ", tmp);
            }
            fputc('\n', emb_file);
        }
    }
    if (num_nodes != num) {
        perror("ERROR: the number of learned embeddings are not equal to num_nodes");
        exit(-1);
    }
    fclose(tmp_emb);
    fclose(emb_file);
    remove(embedding_tmp_filename);
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++)
        if (!strcmp(str, argv[a])) {
            if (a == argc - 1) {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    return -1;
}

real average(const real *arr, int n) {
    real sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum / (real) n;
}

int main(int argc, char **argv) {
    strcpy(network_file, "karate.edgelist");
    sprintf(embedding_file, "%s.emb", network_file);
    strcpy(walk_dir, "walks");
    strcpy(flag, "sp");
    int i;
    if (argc == 1) {
        printf("RiWalk: Fast strucutral node embedding toolkit.\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t--input <file>\n");
        printf("\t\tUse network data from <file> to train the model\n");
        printf("\t--output <file>\n");
        printf("\t\tUse <file> to save the learnt embeddings\n");
        printf("\t--dimensions <int>\n");
        printf("\t\tSet dimension of node embeddings; default is 128\n");
        printf("\t--num-walks <int>\n");
        printf("\t\tNumber of walks per source. Default is 80.\n");
        printf("\t--walk-length <int>\n");
        printf("\t\tLength of walk per source. Default is 10.\n");
        printf("\t--window-size <int>\n");
        printf("\t\tContext size for optimization. Default is 10.\n");
        printf("\t--until-k <int>\n");
        printf("\t\tNeighborhood size k. Default is 4.\n");
        printf("\t--workers <int>\n");
        printf("\t\tUse <int> threads (default 4)\n");
        printf("\t--iter <int>\n");
        printf("\t\tNumber of epochs in SGD. Default is 5.\n");
        printf("\t--flag flag\n");
        printf("\t\tFlag indicating using RiWalk-SP(sp) or RiWalk-WL(wl). Default is sp.\n");
        printf("\t--discount\n");
        printf("\t\tFlag indicating using or not using discount.\n");

        printf("\nExamples:\n");
        printf("./RiWalk --input karate.edgelist --output karate.emb --dimensions 64 --num-walks 80 --walk-length 10 --window-size 5 --until-k 4 --workers 8 --iter 5 --flag sp --discount true\n\n");
        return 0;
    }
    if ((i = ArgPos((char *) "--input", argc, argv)) > 0) strcpy(network_file, argv[i + 1]);
    if ((i = ArgPos((char *) "--output", argc, argv)) > 0) strcpy(embedding_file, argv[i + 1]);
    if ((i = ArgPos((char *) "--dimensions", argc, argv)) > 0) dim = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "--num-walks", argc, argv)) > 0) num_walks = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "--walk-length", argc, argv)) > 0) walk_length = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "--window-size", argc, argv)) > 0) window_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "--until-k", argc, argv)) > 0) until_k = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "--workers", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "--iter", argc, argv)) > 0) iter_num = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "--flag", argc, argv)) > 0) strcpy(flag, argv[i + 1]);
    if ((i = ArgPos((char *) "--discount", argc, argv)) > 0) {
        if (strcmp(argv[i + 1], "true") == 0 || strcmp(argv[i + 1], "True") == 0) {
            discount = 1;
        } else if (strcmp(argv[i + 1], "false") == 0 || strcmp(argv[i + 1], "False") == 0) {
            discount = 0;
        } else {
            printf("Argument for --discount is wrong.\n");
            exit(1);
        }
    }
    printf("--------------------------------\n");
    printf("input: %s\n", network_file);
    printf("output: %s\n", embedding_file);
    printf("dimensions: %d\n", dim);
    printf("num_walks: %d\n", num_walks);
    printf("walk_length: %d\n", walk_length);
    printf("window_size: %d\n", window_size);
    printf("until_k: %d\n", until_k);
    printf("workers: %d\n", num_threads);
    printf("iter: %d\n", iter_num);
    printf("flag: %s\n", flag);
    printf("discount: %d\n", discount);
    printf("--------------------------------\n");

    nodes = (node *) calloc(max_num_nodes, sizeof(node));
    InitHashTable();
    ReadData();
    char tmp[MAX_STRING];
    sprintf(tmp, "rm -rf %s/*", walk_dir);
    system(tmp);
    bfs_times = (real *) malloc(num_threads * sizeof(real));
    ri_times = (real *) malloc(num_threads * sizeof(real));
    walk_times = (real *) malloc(num_threads * sizeof(real));
    RiWalk();
    printf("--------------------------------\n");
    real bfs_time = average(bfs_times, num_threads), ri_time = average(ri_times, num_threads), walk_time = average(
            walk_times, num_threads);
    printf("bfs time %f\n", bfs_time);
    printf("ri time %f\n", ri_time);
    printf("walk time %f\n", walk_time);

    sprintf(tmp, "python src/c_src/w2v.py %s %d %d %d %d %s.tmp", walk_dir, dim,
            window_size, iter_num, num_threads, embedding_file);
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    system(tmp);
    gettimeofday(&stop, NULL);
    real learning_time = (double) (stop.tv_usec - start.tv_usec) / 1000000 + (double) (stop.tv_sec - start.tv_sec);

    SaveEmbeddings();

    strncpy(tmp, embedding_file, strlen(embedding_file) - 4);
    tmp[strlen(embedding_file) - 4] = 0;
    strcat(tmp, "_time.json");
    FILE *f = fopen(tmp, "w");
    if (f == NULL) {
        perror("ERROR: can not open time_json for writing");
        exit(-1);
    }
    fprintf(f, "{\"time\":%f}", bfs_time + ri_time + walk_time + learning_time);
    fclose(f);
    return 0;
}
