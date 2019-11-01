# RiWalk


This repository provides a reference implementation of **RiWalk** as described in the paper:<br>
> RiWalk: Fast Structural Node Embedding via Role Identification.<br>
> Xuewei Ma, Geng Qin, Zhiyang Qiu, Mingxin Zheng, Zhe Wang.<br>
> IEEE International Conference on Data Mining, ICDM, 2019.<br>

The RiWalk algorithm learns continuous representations for nodes in any graph. RiWalk captures structural equivalence between nodes.  

### Prerequisites
RiWalk was written for Python 3. Before to execute RiWalk, it is necessary to install the following packages:
- numpy 
- networkx 
- gensim

### Basic Usage

#### Example
To run RiWalk on Zachary's karate club network using RiWalk-SP, execute the following command from the project home directory:<br/>

	python src/RiWalk.py --input graphs/karate.edgelist --output embs/karate.emb 
	--num-walks 80 --walk-length 10 --window-size 10 --dimensions 128 --until-k 4 --flag sp


#### Full Command List
The full list of command line options is available with 
	
	python src/RiWalk.py --help

#### Input
The supported input format is an edgelist:

	node1_id_int node2_id_int
		

#### Output
The output file has *n+1* lines for a graph with *n* vertices. 
The first line has the following format:

	num_of_nodes dim_of_representation

The next *n* lines are as follows:
	
	node_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation learned by RiWalk.

### Acknowledgements
We would like to thank the authors of [node2vec](https://github.com/aditya-grover/node2vec), [struc2vec](https://github.com/leoribeiro/struc2vec) and [GraphWave](https://github.com/snap-stanford/graphwave) for the open access of the implementations of their methods.

### Miscellaneous
- Please send any questions you might have about the code and/or the algorithm to <xuew.ma@gmail.com>.

- *Note:* This is only a reference implementation of the framework RiWalk.

###  Citation

```
@inproceedings{xuewma2019riwalk,
  title={RiWalk: Fast Structural Node Embedding via Role Identification},
  author={Xuewei Ma, Geng Qin, Zhiyang Qiu, Mingxin Zheng, Zhe Wang},
  booktitle={2019 IEEE International Conference on Data Mining (ICDM)},
  organization={IEEE},
  year={2019}
}
```
