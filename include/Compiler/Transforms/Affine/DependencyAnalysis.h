#ifndef MLIR_TRANSFORMS_DEPENDENCYANALYSIS_H
#define MLIR_TRANSFORMS_DEPENDENCYANALYSIS_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "Compiler/Transforms/Affine/MemoryAccessAnalysis.h"

namespace mlir {

struct LoopCarriedDependency {
  Operation *source;
  Operation *destination;
  Value memref;
  unsigned distance; // distance along the loop
  bool isRAW = false;
  bool isWAR = false;
  bool isWAW = false;
};

class DependencyAnalysis {
public:
  DependencyAnalysis(affine::AffineForOp forOp, const MemoryAccessAnalysis &memAnalysis);

  bool hasDependencies() const { return !dependencies.empty(); }
  const llvm::SmallVector<LoopCarriedDependency, 4> &getDependencies() const {
    return dependencies;
  }

  // Minimum loop-carried dependency distance
  unsigned getMinDependencyDistance() const;

  // Returns true if any dependency distance ≤ threshold
  bool hasTightDependencies(unsigned threshold) const;

  // Maximum chain length of loop-carried dependencies
  unsigned getDependencyChainLength() const;

private:
  void analyzeDependencies();
  void addDependency(Operation *src, Operation *dst);
  bool checkDependency(Operation *src, Operation *dst);
  unsigned computeDistance(Operation *src, Operation *dst);

  affine::AffineForOp forOp;
  const MemoryAccessAnalysis &memAnalysis;
  llvm::SmallVector<LoopCarriedDependency, 4> dependencies;
};

} // namespace mlir

#endif // MLIR_TRANSFORMS_DEPENDENCYANALYSIS_H
