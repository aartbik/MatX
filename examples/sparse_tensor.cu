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

  tensor_t<float, rank> A{{m, n}};
  A.SetVals({{10,0,0}, {0,20,0}, {30,40,0}});
  print(A);
 
  auto As = make_sparse_tensor<float, int, int>(A, COO);
  std::cout << As.str() << " nse=" << As.getNse() << std::endl;
//print(As); // make this work

  MATX_EXIT_HANDLER();
}
