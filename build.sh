#!/usr/bin/bash

set -e

if [ ! -f simdutf.cpp ]; then
    wget https://github.com/simdutf/simdutf/releases/download/v8.0.0/singleheader.zip
    unzip singleheader.zip simdutf.cpp simdutf_c.h simdutf.h
fi

if [ ! -f simdutf.o ]; then
    clang++ -c -march=native -O2 simdutf.cpp
fi
clang -c -march=native -O2 digest.c
clang++ simdutf.o digest.o -o a.out
./a.out
