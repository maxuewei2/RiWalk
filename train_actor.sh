#!/bin/sh

python3 src/RiWalk/RiWalk.py --input graphs/actor.edgelist --output embs/actor.emb --dimensions 128 --num-walks 80 --walk-length 10 --window-size 10 --until-k 4 --workers 4 --iter 5 --flag sp

python3 src/RiWalkRW/RiWalkRW.py --input graphs/actor.edgelist --output embs/actor.emb --dimensions 128 --num-walks 80 --walk-length 10 --window-size 10 --workers 4 --iter 5 --flag sp


gcc -lm -pthread -Ofast -march=native -Wall -ffast-math -Wno-unused-result src/RiWalk-C/RiWalk.c -o src/RiWalk-C/RiWalk
src/RiWalk-C/RiWalk --input graphs/actor.edgelist --output embs/actor.emb --dimensions 128 --num-walks 80 --walk-length 10 --window-size 10 --until-k 4 --workers 10 --iter 5 --flag sp --discount true
src/RiWalk-C/RiWalk --input graphs/actor.edgelist --output embs/actor.emb --dimensions 128 --num-walks 80 --walk-length 10 --window-size 10 --workers 10 --iter 5 --flag rwsp --discount true

