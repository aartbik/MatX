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
  constexpr index_t k = 3;

  tensor_t<float, rank> A{{m, k}};
  tensor_t<float, rank> B{{k, n}};
  tensor_t<float, rank> C{{m, n}};
  A.SetVals({{10, 0, 0}, { 0,20, 0}, {30,40, 0}});
  B.SetVals({{ 1, 2, 3}, { 4, 5, 6}, { 7, 8, 9}});

  (C = matmul(A, B)).run(exec);
  print(C);
 
  auto As = make_sparse_tensor<float, int, int>(A, COO);
  tensor_t<float, rank> C2{{m, n}};
  std::cout << As.str() << " nse=" << As.getNse() << std::endl;
//print(As); // make this work
  (C2 = matmul(As, B)).run(exec);
  print(C2);

  MATX_EXIT_HANDLER();
}
