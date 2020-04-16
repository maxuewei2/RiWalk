#!/bin/sh

graph=actor

echo -------------RiWalk-SP-------------- 
python3 src/RiWalk/RiWalk.py --input graphs/$graph.edgelist --output embs/$graph.emb --dimensions 128 --num-walks 80 --walk-length 10 --window-size 10 --until-k 4 --workers 10 --iter 5 --flag sp

echo \\n------------RiWalk-RW-SP------------
python3 src/RiWalk-RW/RiWalk-RW.py --input graphs/$graph.edgelist --output embs/$graph.emb --dimensions 128 --num-walks 80 --walk-length 10 --window-size 10 --workers 10 --iter 5 --flag sp


gcc -lm -pthread -Ofast -march=native -Wall -ffast-math -Wno-unused-result src/RiWalk-C/RiWalk.c -o src/RiWalk-C/RiWalk
echo \\n------------RiWalk-C-SP-------------
src/RiWalk-C/RiWalk --input graphs/$graph.edgelist --output embs/$graph.emb --dimensions 128 --num-walks 80 --walk-length 10 --window-size 10 --until-k 4 --workers 10 --iter 5 --flag sp --discount true
echo \\n-----------RiWalk-C-RWSP------------
src/RiWalk-C/RiWalk --input graphs/$graph.edgelist --output embs/$graph.emb --dimensions 128 --num-walks 80 --walk-length 10 --window-size 10 --workers 10 --iter 5 --flag rwsp --discount true

