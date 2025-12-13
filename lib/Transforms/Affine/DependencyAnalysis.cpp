#include "Compiler/Transforms/Affine/DependencyAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "Compiler/Transforms/Affine/MemoryAccessAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {

DependencyAnalysis::DependencyAnalysis(affine::AffineForOp forOp,
                                       const MemoryAccessAnalysis &memAnalysis)
    : forOp(forOp), memAnalysis(memAnalysis) {
  analyzeDependencies();
}

void DependencyAnalysis::analyzeDependencies() {
  // Group memory operations by MemRef to avoid O(N^2) comparisons across different buffers
  llvm::DenseMap<Value, llvm::SmallVector<Operation *, 4>> memOpsByRef;

  forOp.walk([&](Operation *op) {
    Value memRef;
    if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op))
      memRef = loadOp.getMemRef();
    else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op))
      memRef = storeOp.getMemRef();
    
    if (memRef)
      memOpsByRef[memRef].push_back(op);
  });

  // Only compare operations that access the same MemRef
  for (auto &entry : memOpsByRef) {
    auto &ops = entry.second;
    for (size_t i = 0; i < ops.size(); ++i) {
      for (size_t j = i + 1; j < ops.size(); ++j) {
        if (checkDependency(ops[i], ops[j]))
          addDependency(ops[i], ops[j]);
      }
    }
  }
}

void DependencyAnalysis::addDependency(Operation *src, Operation *dst) {
  LoopCarriedDependency dep;
  dep.source = src;
  dep.destination = dst;

  dep.distance = computeDistance(src, dst); // single unsigned now

  bool srcIsLoad = isa<affine::AffineLoadOp>(src);
  bool dstIsLoad = isa<affine::AffineLoadOp>(dst);

  dep.isRAW = (!srcIsLoad && dstIsLoad);
  dep.isWAR = (srcIsLoad && !dstIsLoad);
  dep.isWAW = (!srcIsLoad && !dstIsLoad);

  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(src))
    dep.memref = loadOp.getMemRef();
  else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(src))
    dep.memref = storeOp.getMemRef();

  dependencies.push_back(dep);
}

bool DependencyAnalysis::checkDependency(Operation *src, Operation *dst) {
  Value srcMemRef, dstMemRef;

  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(src))
    srcMemRef = loadOp.getMemRef();
  else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(src))
    srcMemRef = storeOp.getMemRef();
  else
    return false;

  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(dst))
    dstMemRef = loadOp.getMemRef();
  else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(dst))
    dstMemRef = storeOp.getMemRef();
  else
    return false;

  if (srcMemRef != dstMemRef)
    return false;

  mlir::affine::MemRefAccess srcAccess(src);
  mlir::affine::MemRefAccess dstAccess(dst);

  unsigned loopDepth = mlir::affine::getNestingDepth(forOp);
  mlir::affine::FlatAffineValueConstraints constraints;
  llvm::SmallVector<mlir::affine::DependenceComponent, 2> components;
  bool allowRAR = false;

  auto depResult = affine::checkMemrefAccessDependence(
      srcAccess, dstAccess, loopDepth, &constraints, &components, allowRAR);

  return depResult.value == mlir::affine::DependenceResult::HasDependence;
}

unsigned DependencyAnalysis::computeDistance(Operation *src, Operation *dst) {
  mlir::affine::MemRefAccess srcAccess(src);
  mlir::affine::MemRefAccess dstAccess(dst);

  unsigned loopDepth = mlir::affine::getNestingDepth(forOp);
  mlir::affine::FlatAffineValueConstraints constraints;
  llvm::SmallVector<mlir::affine::DependenceComponent, 2> components;

  auto depResult = affine::checkMemrefAccessDependence(
      srcAccess, dstAccess, loopDepth, &constraints, &components, false);

  if (depResult.value != mlir::affine::DependenceResult::HasDependence)
    return UINT_MAX;

  // Use the cached memory analysis
  Value memref = nullptr;
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(src))
    memref = loadOp.getMemRef();
  else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(src))
    memref = storeOp.getMemRef();
  
  unsigned stride = 1;
  if (memref) {
      stride = memAnalysis.getStride(memref);
      stride = std::max<unsigned>(1, stride);
  }
  unsigned minDistance = UINT_MAX;
  for (auto &comp : components) {
    if (comp.lb.has_value())
      minDistance = std::min(minDistance, static_cast<unsigned>(std::abs(comp.lb.value())));
    else if (comp.ub.has_value())
      minDistance = std::min(minDistance, static_cast<unsigned>(std::abs(comp.ub.value())));
    else
      minDistance = 1;
  }
  if (minDistance != UINT_MAX)
    llvm:: outs() << "Computed raw minDistance: " << minDistance << "\n";


  return minDistance;
}

unsigned DependencyAnalysis::getMinDependencyDistance() const {
  unsigned minDist = UINT_MAX;
  for (auto &dep : dependencies) {
    minDist = std::min(minDist, dep.distance);
  }
  return minDist == UINT_MAX ? 0 : minDist;
}

unsigned DependencyAnalysis::getDependencyChainLength() const {
  llvm::DenseMap<Operation *, unsigned> chainLength;
  for (auto &dep : dependencies) {
    unsigned srcLen = chainLength.lookup(dep.source);
    chainLength[dep.destination] =
        std::max(chainLength.lookup(dep.destination), srcLen + 1);
  }

  unsigned maxChain = 0;
  for (auto &entry : chainLength)
    maxChain = std::max(maxChain, entry.second);
  return maxChain;
}

bool DependencyAnalysis::hasTightDependencies(unsigned threshold) const {
  for (auto &dep : dependencies)
    if (dep.distance <= threshold)
      return true;
  return false;
}

} // namespace mlir
