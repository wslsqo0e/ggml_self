set -x
CXX_FLAG="-Wall -Werror -Wl,--no-undefined"
gcc ${CXX_FLAG} main_test.cpp ggml.c -o main
