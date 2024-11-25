// sparse tensor prototype

#include <cassert>
#include <cstdio>

#include "matx.h"

using namespace matx;

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();

  cudaStream_t stream = 0;
  cudaExecutor exec{stream};

  constexpr index_t rank = 2;
  constexpr index_t m = 3;
  constexpr index_t n = 3;

  tensor_t<float, rank> a{{m, n}};
  a.SetVals({{10,0,0}, {0,20,0}, {30,40,0}});
  print(a);
  
  constexpr index_t dimRank = rank;
  constexpr index_t lvlRank = rank;
  sparse_tensor_t<float, int, int, dimRank, lvlRank> As(a);
  As.print();

  MATX_EXIT_HANDLER();
}
