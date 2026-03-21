#!/usr/bin/bash

set -e

if [ ! -f simdutf.cpp ]; then
    wget https://github.com/simdutf/simdutf/releases/download/v8.0.0/singleheader.zip
    unzip singleheader.zip simdutf.cpp simdutf_c.h simdutf.h
fi

if [ ! -f simdutf.o ]; then
    clang++ -c -march=native -O2 simdutf.cpp
fi
clang -c -DNDEBUG -march=native -O2 -o digest.perf.o digest.c
clang -c -fsanitize=address -fsanitize=undefined -DTEST -march=native -O2 -o digest.test.o digest.c
clang++ simdutf.o digest.perf.o -o a.out
clang++ -fsanitize=address -fsanitize=undefined simdutf.o digest.test.o -o test
./test
./a.out

llvm-objdump -Mintel --disassemble-symbols=encode_40_simd,decode_40_simd,encode_37,decode_37,encode_37_simd,decode_37_simd > dis
