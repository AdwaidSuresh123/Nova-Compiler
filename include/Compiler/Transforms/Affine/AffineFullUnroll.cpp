// AffineFullUnroll.cpp
#include "AffineFullUnroll.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"


namespace mlir {
namespace nova {

#define GEN_PASS_DEF_AFFINEFULLUNROLL
#include "Compiler/Transforms/Passes.h.inc"

int countLoopBodyOperations(mlir::affine::AffineForOp loop) {
  int opCount = 0;

  loop.walk([&](Operation *op) {
    if (op == loop.getOperation())
      return;
    if (!isa<affine::AffineYieldOp>(op))
      opCount++;
  });
  return opCount;
}

LogicalResult affineLoopUnroll(mlir::affine::AffineForOp forOp){
   std::optional<int64_t> mayBeConstantTripCount =getConstantTripCount(forOp);
   if(mayBeConstantTripCount.has_value()){
      uint64_t tripCount=*mayBeConstantTripCount;
      uint64_t countloopbodyops=countLoopBodyOperations(forOp);
      if (tripCount == 0)
      return success();
      if (tripCount == 1)
       return promoteIfSingleIteration(forOp);
      if (tripCount && tripCount < 16 && countloopbodyops < 20) 
       return mlir::affine::loopUnrollFull(forOp);
      else if (tripCount && tripCount >= 16)
       return mlir::affine::loopUnrollByFactor(forOp, 4);
   }
   return success();
}

// A pass that manually walks the IR
struct AffineFullUnroll : impl::AffineFullUnrollBase<AffineFullUnroll> {
  using AffineFullUnrollBase::AffineFullUnrollBase;

  void runOnOperation() override {
    getOperation()->walk([&](mlir::affine::AffineForOp op) {
      if (failed(affineLoopUnroll(op))) {
        op.emitError("unrolling failed");
        signalPassFailure();
      }
    });
  }
};

} // namespace nova
} // namespace mlir