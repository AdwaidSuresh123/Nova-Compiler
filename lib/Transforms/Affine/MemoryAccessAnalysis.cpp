#include "Compiler/Transforms/Affine/MemoryAccessAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir {

MemoryAccessAnalysis::MemoryAccessAnalysis(affine::AffineForOp forOp)
    : forOp(forOp) {
  analyzeAccessPatterns();
}

void MemoryAccessAnalysis::analyzeAccessPatterns() {
  DenseMap<Value, unsigned> accessCounts;

  // Walk body and collect load/store accesses
  forOp.walk([&](Operation *op) {
    Value memref;
    AffineMap map;
    SmallVector<Value, 4> indices;

    if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
      memref = loadOp.getMemRef();
      map = loadOp.getAffineMap();
      indices.append(loadOp.getMapOperands().begin(),
                     loadOp.getMapOperands().end());
    } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
      memref = storeOp.getMemRef();
      map = storeOp.getAffineMap();
      indices.append(storeOp.getMapOperands().begin(),
                     storeOp.getMapOperands().end());
    } else {
      return;
    }

    accessCounts[memref]++;

    auto it = patterns.find(memref);
    if (it == patterns.end()) {
      MemoryAccessPattern pattern;
      pattern.elementSize = (unsigned)getElementSizeBytes(memref);
      pattern.type = classifyPattern(map, indices, memref);
      pattern.stride = computeStride(map, indices, forOp.getInductionVar(),
                                     memref);
      pattern.memrefValue = memref;
      pattern.numAccesses = 1;
      pattern.hasReuse = false;
      pattern.spatialLocality = 0.0;
      patterns[memref] = pattern;
    } else {
      it->second.numAccesses++;
      it->second.hasReuse = true;
    }
  });

  // Compute spatial locality (weighted by numAccesses)
  for (auto &entry : patterns) {
    entry.second.spatialLocality = computeSpatialLocality(entry.second);
  }
}

MemoryAccessPattern::Type
MemoryAccessAnalysis::classifyPattern(AffineMap map, ArrayRef<Value> indices,
                                      Value memrefValue) {
  if (map.getNumResults() == 0)
    return MemoryAccessPattern::Type::Irregular;

  bool dependsOnIV = false;
  bool isNonLinear = false;
  int64_t bestAbsCoeff = 0;
  Value iv = forOp.getInductionVar();

  for (unsigned r = 0; r < map.getNumResults(); ++r) {
    AffineExpr expr = map.getResult(r);

    expr.walk([&](AffineExpr e) {
      if (auto d = dyn_cast<AffineDimExpr>(e)) {
        unsigned pos = d.getPosition();
        if (pos < indices.size() && indices[pos] == iv)
          dependsOnIV = true;
      }
      if (auto bin = dyn_cast<AffineBinaryOpExpr>(e)) {
        if (bin.getKind() == AffineExprKind::Mul) {
          if (!isa<AffineConstantExpr>(bin.getLHS()) && !isa<AffineConstantExpr>(bin.getRHS()))
            isNonLinear = true;
        }
      }
    });

    for (int pos = 0; pos < (int)indices.size(); ++pos) {
      if (indices[pos] == iv) {
        int64_t coeff = getCoeffForDim(map.getResult(r), (unsigned)pos);
        int64_t mult = (int64_t)linearizationMultiplierForResult(memrefValue, r);
        int64_t contrib = coeff * mult;
        bestAbsCoeff = std::max<int64_t>(bestAbsCoeff, std::abs(contrib));
      }
    }
  }

  if (!dependsOnIV)
    return MemoryAccessPattern::Type::Broadcast;

  if (isNonLinear)
    return MemoryAccessPattern::Type::Irregular;

  if (bestAbsCoeff == 1)
    return MemoryAccessPattern::Type::Sequential;
  if (bestAbsCoeff > 1)
    return MemoryAccessPattern::Type::Strided;

  return MemoryAccessPattern::Type::Irregular;
}
int64_t MemoryAccessAnalysis::getCoeffForDim(AffineExpr expr,
                                             unsigned dimPos) const {
  if (!expr)
    return 0;

  if (auto c = dyn_cast<AffineConstantExpr>(expr))
    return 0;

  if (auto d = dyn_cast<AffineDimExpr>(expr))
    return (d.getPosition() == (unsigned)dimPos) ? 1 : 0;

  if (auto bin = dyn_cast<AffineBinaryOpExpr>(expr)) {
    AffineExpr lhs = bin.getLHS();
    AffineExpr rhs = bin.getRHS();
    switch (bin.getKind()) {
    case AffineExprKind::Add:
      return getCoeffForDim(lhs, dimPos) + getCoeffForDim(rhs, dimPos);
    case AffineExprKind::Mul:
      if (auto lc = dyn_cast<AffineConstantExpr>(lhs))
        return lc.getValue() * getCoeffForDim(rhs, dimPos);
      if (auto rc = dyn_cast<AffineConstantExpr>(rhs))
        return rc.getValue() * getCoeffForDim(lhs, dimPos);
      return 0;
    default:
      return 0;
    }
  }
  return 0;
}

int64_t MemoryAccessAnalysis::computeStride(AffineMap map,
                                            ArrayRef<Value> indices,
                                            Value inductionVar,
                                            Value memrefValue) {
  if (map.getNumResults() == 0)
    return 0;

  int64_t elementDeltaPerIV = 0;
  for (unsigned r = 0; r < map.getNumResults(); ++r) {
    AffineExpr expr = map.getResult(r);
    for (unsigned pos = 0; pos < (unsigned)indices.size(); ++pos) {
      if (indices[pos] == inductionVar) {
        int64_t coeff = getCoeffForDim(expr, pos);
        int64_t mult = (int64_t)linearizationMultiplierForResult(memrefValue, r);
        elementDeltaPerIV += coeff * mult;
      }
    }
  }
  return std::abs(elementDeltaPerIV);
}

uint64_t MemoryAccessAnalysis::getElementSizeBytes(Value memrefValue) const {
  if (auto memrefType = dyn_cast<MemRefType>(memrefValue.getType())) {
    Type elt = memrefType.getElementType();
    if (auto intTy = dyn_cast<IntegerType>(elt))
      return intTy.getWidth() / 8u;
    if (auto floatTy = dyn_cast<FloatType>(elt))
      return floatTy.getWidth() / 8u;
    if (auto vecTy = dyn_cast<VectorType>(elt)) {
      Type elementType = vecTy.getElementType();
      uint64_t elementBytes = 0;
      if (auto intElem = dyn_cast<IntegerType>(elementType))
        elementBytes = intElem.getWidth() / 8u;
      else if (auto floatElem = dyn_cast<FloatType>(elementType))
        elementBytes = floatElem.getWidth() / 8u;
      else
        elementBytes = 4; 
      uint64_t numElems = 1;
      for (int64_t d : vecTy.getShape())
        numElems *= d;
      return numElems * elementBytes;
    }
    return 4; // fallback for unknown types
  }
  return 4;
}


uint64_t MemoryAccessAnalysis::linearizationMultiplierForResult(
    Value memrefValue, unsigned resultIndex) const {
  auto memrefTy = dyn_cast_or_null<MemRefType>(memrefValue.getType());
  if (!memrefTy)
    return 1;

  ArrayRef<int64_t> shape = memrefTy.getShape();
  unsigned rank = (unsigned)shape.size();
  if (rank == 0)
    return 1;
  if (resultIndex >= rank)
    return 1;

  uint64_t mult = 1;
  for (unsigned r = resultIndex + 1; r < rank; ++r) {
    int64_t s = shape[r];
    if (s <= 0)
      return 1;
    mult *= (uint64_t)s;
  }
  return mult;
}

double MemoryAccessAnalysis::computeSpatialLocality(
    const MemoryAccessPattern &pattern) const {
  if (pattern.type == MemoryAccessPattern::Type::Sequential)
    return 1.0;
  if (pattern.type == MemoryAccessPattern::Type::Broadcast)
    return 1.0;
  if (pattern.type == MemoryAccessPattern::Type::Strided) {
    int64_t s = pattern.stride;
    if (s <= 1)
      return 1.0;
    double val = 1.0 / (1.0 + 0.25 * (double)(s - 1));
    if (val < 0.0)
      val = 0.0;
    if (val > 1.0)
      val = 1.0;
    return val;
  }
  return 0.0;
}

bool MemoryAccessAnalysis::hasGoodDataLocality() const {
  if (patterns.empty())
    return false;

  double weighted = 0.0;
  uint64_t totalAccesses = 0;
  for (auto &entry : patterns) {
    weighted += entry.second.spatialLocality * (double)entry.second.numAccesses;
    totalAccesses += entry.second.numAccesses;
  }
  if (totalAccesses == 0)
    return false;
  double avg = weighted / (double)totalAccesses;
  return avg > 0.7;
}

unsigned MemoryAccessAnalysis::getReuseDistance(Value memref) const {
  auto it = patterns.find(memref);
  if (it == patterns.end() || !it->second.hasReuse)
    return UINT_MAX;

  if (it->second.stride == 0)
    return 1;
  return 1;
}

unsigned MemoryAccessAnalysis::estimateCacheFootprint() const {
  constexpr uint64_t CACHE_LINE_SIZE = 64;
  uint64_t totalFootprintBytes = 0;

  for (auto &entry : patterns) {
    const MemoryAccessPattern &p = entry.second;
    Value memref = p.memrefValue;
    if (!memref)
      continue;

    uint64_t elementBytes = getElementSizeBytes(memref);
    if (elementBytes == 0)
      elementBytes = 4; 


    uint64_t strideBytes = (uint64_t)p.stride * elementBytes;

    uint64_t numAccesses = p.numAccesses;
    uint64_t rangeBytes = 0;
    if (numAccesses == 1)
      rangeBytes = elementBytes;
    else if (strideBytes > 0)
      rangeBytes = (numAccesses - 1) * strideBytes + elementBytes;
    else
      rangeBytes = numAccesses * elementBytes;

    uint64_t gcdStep = std::gcd<uint64_t>(
        std::max<uint64_t>(1, strideBytes), elementBytes);
    uint64_t uniqueElems = (rangeBytes + gcdStep - 1) / gcdStep;
    uint64_t uniqueBytes = uniqueElems * gcdStep;

    uint64_t cacheLines =
        (uniqueBytes + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE;
    uint64_t footprintBytes = cacheLines * CACHE_LINE_SIZE;

    totalFootprintBytes += footprintBytes;
  }

  if (totalFootprintBytes > UINT_MAX)
    return UINT_MAX;

  return static_cast<unsigned>(totalFootprintBytes);
}


bool MemoryAccessAnalysis::benefitsFromUnrollAndJam() const {
  return hasGoodDataLocality() && estimateCacheFootprint() < 32 * 1024;
}

} // namespace mlir