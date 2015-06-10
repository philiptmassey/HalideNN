#!/bin/bash

g++ utils.cpp layers.cpp main.cpp -g -I ../include -L ../bin -lHalide `libpng-config --cflags --ldflags` -std=c++11
DYLD_LIBRARY_PATH=../bin ./a.out
