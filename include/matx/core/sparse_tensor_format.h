#pragma once

namespace matx {

//
// MatX uses a single versatile sparse tensor type that uses a tensor format
// DSL (Domain Specific Language) to describe a vast space of storage formats
// Although the tensor format can easily define many common storage formats
// (such as Dense, CSR, CSC, BSR), it can also define many less common storage
// formats. In addition, the tensor format DSL can be extended to include even
// more storage formats in the future.
//
// In the tensor format, the term **dimension** is used to refer to the axes of
// the semantic tensor (as seen by the user), and the term **level** to refer to
// the axes of the actual storage format (how it eventually resides in memory).
//
// The tensor format contains a map that provides the following:
//
// (1) An ordered sequence of dimension specifications, each of which includes:
//
//     (*) a dimension-expression, which provides a reference to each dimension
//
// (2) An ordered sequence of level specifications, each of which includes:
//
//     (*) a level expression, which defines what is stored in each level
//     (*) a required level type, which defines how the level is stored,
//     including:
//         (+) a required level format
//         (+) a collection of level properties
//
// Currently, the following level formats are supported:
//
// (1) dense: level is dense, entries along the level are stored and linearized
// (2) compressed: level is sparse, only nonzeros along the level are stored
//     with the compact positions and coordinates encoding
// (3) singleton: a variant of the compressed format, for when coordinates have
//     no siblings
//
// All level formats have the following level properties:
//
// (1) non/unique (are duplicates allowed at that level),
// (2) un/ordered (are coordinates sorted at that level).
//
// Examples:
//
// COO:
//     map = (i, j) -> ( i : compressed(non-unique), j : singleton )
//
// CSR:
//     map = (i, j) -> ( i : dense, j : compressed )
//
// CSC:
//     map = (i, j) -> ( j : dense, i : compressed )
//
// BSR with 2x3 blocks:
//      map = ( i, j ) -> ( i floordiv 2 : dense,
//                          j floordiv 3 : compressed,
//                          i mod 2      : dense,
//                          j mod 3      : dense )
//
// The idea of a single versatile sparse tensor type has its roots in
// sparse compilers, first pioneered for sparse linear algebra in [Bik96]
// and formalized to sparse tensor algebra in [Kjolstad20]. The generalization
// to higher-dimensional levels was introduced in [MLIR22].
//
// [Bik96] Aart J.C. Bik. Compiler Support for Sparse Matrix Computations.
//         PhD thesis, Leiden University, May 1996.
// [Kjolstad20] Fredrik Berg Kjolstad. Sparse Tensor Algebra Compilation.
//              PhD thesis, MIT, February, 2020.
// [MLIR22] Aart J.C. Bik, Penporn Koanantakool, Tatiana Shpeisman,
//        Nicolas Vasilache, Bixia Zheng, and Fredrik Kjolstad.
//        Compiler Support for Sparse Tensor Computations in MLIR.
//        ACM Transactions on Architecture and Code Optimization, June, 2022.
//

//
// A level type consists of a level format together with a set of
// level properties (ordered and unique by default).
// TODO: split out type and properties, generalize
//
enum class LvlType { Dense, Singleton, Compressed, CompressedNonUnique };

//
// A level expression consists of an expression in terms of dimension
// variables. Currently, the following expressions are supported:
//
// (1) di          (cj == 1)
// (2) di * 2
// (3) di div 2    (floor-div)
// (4) di mod 2
//
class LvlExpr {
public:
  enum LvlOp { Mul, Div, Mod };
  constexpr LvlExpr(int d) : op(Mul), di(d), cj(1) {}
  constexpr LvlExpr(int d, LvlOp o, int c) : op(o), di(d), cj(c) {}
  const LvlOp op;
  const int di;
  const int cj;
};

//
// A level specification consists of a level expression and a level type.
//
class LvlSpec {
public:
  constexpr LvlSpec(LvlExpr e, LvlType t) : exp(e), typ(t) {}
  const LvlExpr exp;
  const LvlType typ;
};

//
// A tensor format consists of an implicit ordered sequence of dimension
// specifications (d0, d1, etc.) and an explicit ordered sequence of level
// specifications (e.g. d0 : Dense, d1 : Compressed).
//
template <int DIM, int LVL> class TensorFormat {
public:
  template <typename... T> constexpr TensorFormat(T... t) : lvlSpecs{t...} {}

  // Get spec at given level.
  LvlSpec getLvlSpec(int l) const { return lvlSpecs[l]; }

  // Translate tensor dimensions to levels.
  template <typename CRD>
  CRD *dim2lvl(const CRD *dims, CRD *lvls, bool asSize) const;

  // Translate tensor levels to dimensions.
  template <typename CRD> CRD *lvl2dim(const CRD *lvls, CRD *dims) const;

  // Debugging.
  void print() const;

private:
  const LvlSpec lvlSpecs[LVL];
};

template <int DIM, int LVL>
template <typename CRD>
CRD *TensorFormat<DIM, LVL>::dim2lvl(const CRD *dims, CRD *lvls,
                                     bool asSize) const {
  for (int l = 0; l < LVL; l++) {
    const LvlSpec &spec = lvlSpecs[l];
    switch (spec.exp.op) {
    case LvlExpr::Mul:
      lvls[l] = (dims[spec.exp.di] * spec.exp.cj);
      break;
    case LvlExpr::Div:
      lvls[l] = (dims[spec.exp.di] / spec.exp.cj);
      break;
    case LvlExpr::Mod:
      lvls[l] = asSize ? spec.exp.cj : (dims[spec.exp.di] % spec.exp.cj);
      break;
    }
  }
  return lvls;
}

template <int DIM, int LVL> void TensorFormat<DIM, LVL>::print() const {
  std::cout << "(";
  for (int d = 0; d < DIM; d++) {
    std::cout << " d" << d;
    if (d != DIM - 1)
      std::cout << ",";
  }
  std::cout << " ) -> (";
  for (int l = 0; l < LVL; l++) {
    const LvlSpec &spec = lvlSpecs[l];
    std::cout << " d" << spec.exp.di;
    switch (spec.exp.op) {
    case LvlExpr::Mul:
      if (spec.exp.cj != 1) {
        std::cout << " * " << spec.exp.cj;
      }
      break;
    case LvlExpr::Div:
      std::cout << " div " << spec.exp.cj;
      break;
    case LvlExpr::Mod:
      std::cout << " mod " << spec.exp.cj;
      break;
    }
    std::cout << " : ";
    switch (spec.typ) {
    case LvlType::Dense:
      std::cout << "dense";
      break;
    case LvlType::Singleton:
      std::cout << "singleton";
      break;
    case LvlType::Compressed:
      std::cout << "compressed";
      break;
    case LvlType::CompressedNonUnique:
      std::cout << "compressed(non-unique)";
      break;
    }
    if (l != LVL - 1)
      std::cout << ",";
  }
  std::cout << " )" << std::endl;
};

//
// Predefined common tensor formats. Note that even though the tensor format
// was introduced to define a single versatile sparse tensor type, the
// "all-dense" format also naturally describes dense scalars, vectors,
// matrices, and tensors, with all d-major format variants.
//

// Scalars.
constexpr auto Scalar = TensorFormat<0, 0>({});

// Vectors.
constexpr auto DnVec = TensorFormat<1, 1>(LvlSpec(LvlExpr(0), LvlType::Dense));
constexpr auto SpVec =
    TensorFormat<1, 1>(LvlSpec(LvlExpr(0), LvlType::Compressed));

// LvlType::Dense Matrices.
constexpr auto DnMat = TensorFormat<2, 2>(LvlSpec(LvlExpr(0), LvlType::Dense),
                                          LvlSpec(LvlExpr(1), LvlType::Dense));
constexpr auto DnMatCol = TensorFormat<2, 2>(
    LvlSpec(LvlExpr(1), LvlType::Dense), LvlSpec(LvlExpr(0), LvlType::Dense));

// Sparse Matrices.
constexpr auto COO =
    TensorFormat<2, 2>(LvlSpec(LvlExpr(0), LvlType::CompressedNonUnique),
                       LvlSpec(LvlExpr(1), LvlType::Singleton));
constexpr auto CSR =
    TensorFormat<2, 2>(LvlSpec(LvlExpr(0), LvlType::Dense),
                       LvlSpec(LvlExpr(1), LvlType::Compressed));
constexpr auto CSC =
    TensorFormat<2, 2>(LvlSpec(LvlExpr(1), LvlType::Dense),
                       LvlSpec(LvlExpr(0), LvlType::Compressed));
constexpr auto DCSR =
    TensorFormat<2, 2>(LvlSpec(LvlExpr(0), LvlType::Compressed),
                       LvlSpec(LvlExpr(1), LvlType::Compressed));
constexpr auto DCSC =
    TensorFormat<2, 2>(LvlSpec(LvlExpr(1), LvlType::Compressed),
                       LvlSpec(LvlExpr(0), LvlType::Compressed));
constexpr auto CROW =
    TensorFormat<2, 2>(LvlSpec(LvlExpr(0), LvlType::Compressed),
                       LvlSpec(LvlExpr(1), LvlType::Dense));
constexpr auto CCOL =
    TensorFormat<2, 2>(LvlSpec(LvlExpr(1), LvlType::Compressed),
                       LvlSpec(LvlExpr(0), LvlType::Dense));

// Sparse Block Matrices.
constexpr auto BSR(int m, int n) {
  return TensorFormat<2, 4>(
      LvlSpec(LvlExpr(0, LvlExpr::Div, m), LvlType::Dense),
      LvlSpec(LvlExpr(1, LvlExpr::Div, n), LvlType::Compressed),
      LvlSpec(LvlExpr(0, LvlExpr::Mod, m), LvlType::Dense),
      LvlSpec(LvlExpr(1, LvlExpr::Mod, n), LvlType::Dense));
}
constexpr auto BSRCol(int m, int n) {
  return TensorFormat<2, 4>(
      LvlSpec(LvlExpr(0, LvlExpr::Div, m), LvlType::Dense),
      LvlSpec(LvlExpr(1, LvlExpr::Div, n), LvlType::Compressed),
      LvlSpec(LvlExpr(1, LvlExpr::Mod, n), LvlType::Dense),
      LvlSpec(LvlExpr(0, LvlExpr::Mod, m), LvlType::Dense));
}

// 3-D Tensors.
constexpr auto COO3 =
    TensorFormat<3, 3>(LvlSpec(LvlExpr(0), LvlType::CompressedNonUnique),
                       LvlSpec(LvlExpr(1), LvlType::Singleton),
                       LvlSpec(LvlExpr(2), LvlType::Singleton));

// 4-D Tensors.
constexpr auto COO4 =
    TensorFormat<4, 4>(LvlSpec(LvlExpr(0), LvlType::CompressedNonUnique),
                       LvlSpec(LvlExpr(1), LvlType::Singleton),
                       LvlSpec(LvlExpr(2), LvlType::Singleton),
                       LvlSpec(LvlExpr(3), LvlType::Singleton));

// 5-D Tensors.
constexpr auto COO5 =
    TensorFormat<5, 5>(LvlSpec(LvlExpr(0), LvlType::CompressedNonUnique),
                       LvlSpec(LvlExpr(1), LvlType::Singleton),
                       LvlSpec(LvlExpr(2), LvlType::Singleton),
                       LvlSpec(LvlExpr(3), LvlType::Singleton),
                       LvlSpec(LvlExpr(4), LvlType::Singleton));

} // namespace matx
