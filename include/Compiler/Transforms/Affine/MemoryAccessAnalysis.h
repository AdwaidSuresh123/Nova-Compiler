#ifndef MLIR_TRANSFORMS_MEMORYACCESSANALYSIS_H
#define MLIR_TRANSFORMS_MEMORYACCESSANALYSIS_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Value.h"

namespace mlir {

struct MemoryAccessPattern {
  enum class Type { Sequential, Strided, Broadcast, Irregular };
  unsigned elementSize;
  Type type;
  int64_t stride;
  mlir::Value memrefValue;
  unsigned numAccesses;
  bool hasReuse;
  double spatialLocality;
};

class MemoryAccessAnalysis {
public:
  explicit MemoryAccessAnalysis(affine::AffineForOp forOp);
  void analyzeAccessPatterns();
  MemoryAccessPattern::Type classifyPattern(AffineMap map,
                                            ArrayRef<Value> indices,
                                            Value memrefValue);
  int64_t getCoeffForDim(AffineExpr expr, unsigned dimPos) const;
  int64_t computeStride(AffineMap map, ArrayRef<Value> indices,
                        Value inductionVar, Value memrefValue);
  uint64_t getElementSizeBytes(Value memrefValue) const;
  uint64_t linearizationMultiplierForResult(Value memrefValue,
                                            unsigned resultIndex) const;
  double computeSpatialLocality(const MemoryAccessPattern &pattern) const;
  bool hasGoodDataLocality() const;
  unsigned getReuseDistance(Value memref) const;
  unsigned estimateCacheFootprint() const;
  bool benefitsFromUnrollAndJam() const;
  const llvm::DenseMap<Value, MemoryAccessPattern> &getPatterns() const {
    return patterns;
  }
  unsigned getStride(Value memref) const {
    auto it = patterns.find(memref);
    if (it != patterns.end())
      return it->second.stride;
    return 1; // default stride
  }

private:
  affine::AffineForOp forOp;
  llvm::DenseMap<Value, MemoryAccessPattern> patterns;
};

} 

#endif


