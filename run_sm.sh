set -x
CXX_FLAG="-Wall -Werror -Wl,--no-undefined"
g++ ${CXX_FLAG} main_test.cpp ggml.c -o main
