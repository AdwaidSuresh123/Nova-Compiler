#ifndef MLIR_TRANSFORMS_DEPENDENCYANALYSISTESTPASS_H
#define MLIR_TRANSFORMS_DEPENDENCYANALYSISTESTPASS_H

#include "Compiler/Transforms/Affine/DependencyAnalysis.h"
#include "Compiler/Transforms/Affine/MemoryAccessAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <string>
#include "llvm/ADT/DenseMap.h"

namespace mlir {


struct UnrollDecision {
    bool shouldUnroll;
    unsigned factor;
    bool useUnrollAndJam;
    std::string reason;
};


class DependencyAnalysisTestPass
    : public PassWrapper<DependencyAnalysisTestPass, OperationPass<func::FuncOp>> {

public:
    StringRef getArgument() const override {
        return "test-dependency-analysis";
    }

    StringRef getDescription() const override {
        return "Test pass for loop dependency analysis";
    }

    void runOnOperation() override;
    void analyzeLoop(affine::AffineForOp forOp);
    UnrollDecision makeUnrollDecision(affine::AffineForOp forOp,
                                      const DependencyAnalysis &depAnalysis,
                                      const MemoryAccessAnalysis &memAnalysis);

    unsigned estimateCodeSize(affine::AffineForOp forOp);
    unsigned estimateUnrolledSize(affine::AffineForOp forOp, unsigned factor);

    bool performUnroll(affine::AffineForOp forOp, unsigned factor);
    bool performUnrollAndJam(affine::AffineForOp forOp, unsigned factor);

private:
    llvm::DenseMap<Value, UnrollDecision> decision;
};

void registerDependencyAnalysisTestPass();
} // namespace mlir

#endif // MLIR_TRANSFORMS_DEPENDENCYANALYSISTESTPASS_H