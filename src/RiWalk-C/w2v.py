from gensim.models import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
import glob
import sys
import logging
import time


class Sentences(object):
    """
    a wrapper of random walk files to feed to word2vec
    """
    def __init__(self, file_names):
        self.file_names = file_names

    def __iter__(self):
        fs = []
        for file_name in self.file_names:
            fs.append(open(file_name,'r'))
        while True:
            flag = 0
            for i, f in enumerate(fs):
                line = f.readline().strip()
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


def learn_embeddings(walk_dir,dim,window_size,iter_num,workers,embedding_filename):
    """
    learn embeddings from random walks.
    hs:  0:negative sampling 1:hierarchica softmax
    sg:  0:CBOW              1:skip-gram
    """
    logging.debug('begin learning embeddings')
    learning_begin_time = time.time()

    walk_files = glob.glob('%s/__random_walks_*.txt' % walk_dir)
    sentences = Sentences(walk_files)
    model = Word2Vec(sentences, size=dim, window=window_size, min_count=0, sg=1, hs=0, workers=workers, iter=iter_num)

    learning_end_time = time.time()
    logging.debug('done learning embeddings')
    logging.debug('learning time: {}'.format(learning_end_time - learning_begin_time))
    print('learning_time', learning_end_time - learning_begin_time, flush=True)
    model.wv.save_word2vec_format(fname=embedding_filename, binary=False)
    return model.wv
    
    
if __name__=='__main__':
    print(sys.argv)
    walk_dir,dim,window_size,iter_num,workers,embedding_filename=sys.argv[1:]
    dim=int(dim)
    window_size=int(window_size)
    iter_num=int(iter_num)
    workers=int(workers)
    learn_embeddings(walk_dir,dim,window_size,iter_num,workers,embedding_filename)
    