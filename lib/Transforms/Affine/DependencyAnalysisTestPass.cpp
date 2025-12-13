#include "Compiler/Transforms/Affine/DependencyAnalysisTestPass.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

// Configuration constants - more aggressive for matrix operations
constexpr unsigned MAX_CODE_SIZE = 200;
constexpr unsigned L1_CACHE_SIZE = 32 * 1024;
constexpr unsigned MAX_UNROLL_FACTOR = 16;
constexpr unsigned INNER_LOOP_MAX_FACTOR = 8;
constexpr unsigned MID_LOOP_MAX_FACTOR = 4;

// Helper to check if a loop has reduction pattern (accumulation)
static bool isReductionPattern(affine::AffineForOp forOp) {
  bool hasReadWrite = false;
  
  forOp.walk([&](affine::AffineStoreOp storeOp) {
    Value memRef = storeOp.getMemRef();
    AffineMap storeMap = storeOp.getAffineMap();
    
    // Check if there's a load from the same location
    forOp.walk([&](affine::AffineLoadOp loadOp) {
      if (loadOp.getMemRef() == memRef) {
        // Check if accessing same indices (reduction pattern)
        if (loadOp.getAffineMap() == storeMap) {
          // Verify operands match
          bool sameOperands = true;
          if (loadOp.getMapOperands().size() == storeOp.getMapOperands().size()) {
            for (size_t i = 0; i < loadOp.getMapOperands().size(); ++i) {
              if (loadOp.getMapOperands()[i] != storeOp.getMapOperands()[i]) {
                sameOperands = false;
                break;
              }
            }
            if (sameOperands) {
              hasReadWrite = true;
            }
          }
        }
      }
    });
  });
  
  return hasReadWrite;
}

// Check if loop is innermost
static bool isInnermostLoop(affine::AffineForOp forOp) {
  bool hasNestedLoop = false;
  forOp.walk([&](affine::AffineForOp nested) {
    if (nested != forOp) {
      hasNestedLoop = true;
    }
  });
  return !hasNestedLoop;
}

// Count nesting depth
static unsigned getNestingDepth(affine::AffineForOp forOp) {
  unsigned depth = 0;
  Operation *parent = forOp->getParentOp();
  while (parent) {
    if (isa<affine::AffineForOp>(parent)) depth++;
    parent = parent->getParentOp();
  }
  return depth;
}

bool isValidOperation(mlir::affine::AffineForOp forOp) {
    return forOp && forOp.getOperation()->getBlock();
}

UnrollDecision DependencyAnalysisTestPass::makeUnrollDecision(
    affine::AffineForOp forOp,
    const DependencyAnalysis &depAnalysis,
    const MemoryAccessAnalysis &memAnalysis) {

  UnrollDecision decision{false, 1, false, ""};

  unsigned nestingDepth = getNestingDepth(forOp);
  bool isInnermost = isInnermostLoop(forOp);
  bool isReduction = isReductionPattern(forOp);
  unsigned baseCodeSize = estimateCodeSize(forOp);

  std::vector<unsigned> candidateFactors = {16, 8, 4, 2};
  std::optional<int64_t> tripCount = getConstantTripCount(forOp);

  unsigned baseFootprint = memAnalysis.estimateCacheFootprint();

  llvm::outs() << "  Nesting depth: " << nestingDepth 
               << ", Innermost: " << isInnermost
               << ", Reduction: " << isReduction;
  
  if (tripCount.has_value()) {
    llvm::outs() << ", trip count: " << *tripCount;
  }
  llvm::outs() << "\n";
 
  // Determine max factor based on position in nest
  unsigned maxAllowedFactor = MAX_UNROLL_FACTOR;
  
  if (nestingDepth == 0) {
    // Check if loop body exists and is valid
    Block* body = forOp.getBody();
    if (!body || body->empty()) {
      // Empty loop - full unroll
      decision = {true, tripCount.value_or(1), false, "Empty loop - full unroll"};
      return decision;
    }
    
    // Check if this is a 1D loop (no nested affine for loops in the body)
    bool is1DLoop = body->getOps<affine::AffineForOp>().empty() &&  !forOp->getParentOfType<affine::AffineForOp>() ;
    
    llvm::outs() << "  Is 1D loop: " << (is1DLoop ? "Yes" : "No") << "\n";
    
    if (is1DLoop) {
      unsigned localMaxFactor = 1;
      
      if (tripCount.has_value() && 
          estimateUnrolledSize(forOp, *tripCount) < 200 && 
          (baseFootprint * (*tripCount)) < L1_CACHE_SIZE) {
        localMaxFactor = *tripCount;
      } else {
        for (unsigned f : candidateFactors) {
          if ((estimateUnrolledSize(forOp, f) < 200) && 
              (baseFootprint * f) < L1_CACHE_SIZE) {
            localMaxFactor = f;
            break; 
          }
        }
      }
      
      std::string reason = "Single loop detected, unroll factor set to " + 
                         std::to_string(localMaxFactor);
      decision = {true, localMaxFactor, false, reason};
      return decision;
    } else {
      decision.reason = "Outermost loop but not 1D, skipping unroll";
      return decision;
    }
  }

  if (nestingDepth == 1) {  // Middle loop  
    maxAllowedFactor = std::min(maxAllowedFactor, 8u);
  }

  if (nestingDepth >= 2 && isInnermost) {  // Innermost loop
    maxAllowedFactor = 16;
  }
  
  // For small trip counts, adjust max factor
  if (tripCount.has_value()) {
    llvm::outs() << "  Trip count: " << *tripCount << "\n";
    if (*tripCount <= 4) {
      maxAllowedFactor = std::min(maxAllowedFactor, 
                                   static_cast<unsigned>(*tripCount));
    }
  }
  
  llvm::outs() << "  Base code size: " << baseCodeSize << "\n";

  // More lenient code size check
  if (baseCodeSize > 400) {
    maxAllowedFactor = std::min(maxAllowedFactor, 2u);
    llvm::outs() << "  Large base size, limiting to 2\n";
  }

  // Find optimal unroll factor
  unsigned selectedFactor = 1;

  for (unsigned f : candidateFactors) {
    if (f > maxAllowedFactor) continue;
    if (tripCount.has_value() && f > *tripCount) continue;

    unsigned estimatedSize = estimateUnrolledSize(forOp, f);
    llvm::outs() << "  Factor " << f << " -> size: " << estimatedSize << "\n";

    if (estimatedSize <= MAX_CODE_SIZE) {
      selectedFactor = f;
      break;
    }
  }

  if (selectedFactor == 1) {
    decision.reason = "No beneficial unroll factor found (code size)";
    return decision;
  }
  
  unsigned expandedFootprint = baseFootprint * selectedFactor;
  llvm::outs() << "  Cache footprint: " << baseFootprint 
               << " -> " << expandedFootprint << " bytes\n";

  if (expandedFootprint > L1_CACHE_SIZE) {
    // Scale back factor
    while (selectedFactor > 2 && 
           baseFootprint * selectedFactor > L1_CACHE_SIZE) {
      selectedFactor /= 2;
    }
    llvm::outs() << "  Reduced factor to " << selectedFactor 
                 << " for cache\n";
  }

  // Dependency check - but ignore reduction patterns
  if (!isReduction) {
    unsigned minDepDist = depAnalysis.getMinDependencyDistance();
    if (minDepDist < UINT_MAX && minDepDist > 0) {
      if (minDepDist < selectedFactor) {
        selectedFactor = std::max(1u, minDepDist - 1);
        llvm::outs() << "  Dependency limits factor to " 
                     << selectedFactor << "\n";
      }
    }
  } else {
    llvm::outs() << "  Reduction pattern: allowing full unroll despite deps\n";
  }

  if (selectedFactor <= 1) {
    decision.reason = "No profitable factor after all constraints";
    return decision;
  }

  // Align to power of 2
  if (selectedFactor > 1 && (selectedFactor & (selectedFactor - 1)) != 0) {
    unsigned powerOf2 = 1;
    while (powerOf2 * 2 <= selectedFactor) {
      powerOf2 *= 2;
    }
    selectedFactor = powerOf2;
    llvm::outs() << "  Rounded to power-of-2: " << selectedFactor << "\n";
  }

  // Check for unroll-and-jam opportunity
  bool canJam = false;
  auto outerFor = forOp->getParentOfType<affine::AffineForOp>();
  if (outerFor && nestingDepth == 1 && !isInnermost) {
    canJam = true;
  }

  if (canJam && 
      memAnalysis.benefitsFromUnrollAndJam() && 
      selectedFactor >= 4) {
    decision = {true, selectedFactor, true,
                "Unroll-and-jam for data locality"};
    return decision;
  }

  std::string reason = "Unroll factor " + std::to_string(selectedFactor) +
                      " (depth=" + std::to_string(nestingDepth) + 
                      (isReduction ? ", reduction" : "") + ")";
  decision = {true, selectedFactor, false, reason};

  return decision;
}

unsigned DependencyAnalysisTestPass::estimateCodeSize(
    affine::AffineForOp forOp) {

  unsigned size = 0;
  unsigned memOps = 0;
  unsigned arithOps = 0;
  unsigned affineOps = 0;

  forOp.walk([&](Operation *op) {
    if (isa<affine::AffineLoadOp, affine::AffineStoreOp>(op)) {
      size += 1;
      memOps++;
    } else if (isa<arith::MulIOp, arith::MulFOp>(op)) {
      size += 1;
      arithOps++;
    } else if (isa<arith::DivSIOp, arith::DivUIOp, arith::DivFOp>(op)) {
      size += 2;
      arithOps++;
    } else if (isa<arith::AddIOp, arith::AddFOp, 
                   arith::SubIOp, arith::SubFOp>(op)) {
      size += 1;
      arithOps++;
    } else if (isa<affine::AffineApplyOp>(op)) {
      size += 1;
      affineOps++;
    } else {
      size += 1;
    }
  });

  // Less aggressive penalty
  if (memOps > 32) {
    size += (memOps - 32) / 2;
  }

  llvm::outs() << "    Code details: MemOps=" << memOps
               << ", ArithOps=" << arithOps 
               << ", AffineOps=" << affineOps << "\n";

  return size;
}

unsigned DependencyAnalysisTestPass::estimateUnrolledSize(
    affine::AffineForOp forOp, unsigned factor) {

  unsigned baseSize = estimateCodeSize(forOp);
  unsigned overhead = 4; // Loop control overhead
  
  // Unrolled body size
  unsigned unrolledBody = baseSize * factor;
  
  // Overhead amortization
  unsigned savedOverhead = overhead * (factor - 1) / 2;
  
  // Register pressure penalty (less aggressive)
  unsigned regPressure = 0;
  if (factor >= 16) {
    regPressure = baseSize / 3;
  } else if (factor >= 8) {
    regPressure = baseSize / 6;
  }
  
  unsigned finalSize = unrolledBody;
  if (finalSize > savedOverhead) {
    finalSize -= savedOverhead;
  }
  finalSize += regPressure;

  return finalSize;
}

void DependencyAnalysisTestPass::analyzeLoop(affine::AffineForOp forOp) {
  // Verify operation is still valid
  if (!isValidOperation(forOp)) {
    return;
  }

  llvm::outs() << "\n=== Analyzing loop at " << forOp.getLoc() << " ===\n";

  MemoryAccessAnalysis memAnalysis(forOp);
  DependencyAnalysis depAnalysis(forOp, memAnalysis);

  // Check if this is a reduction pattern
  bool isReduction = isReductionPattern(forOp);
  


  // Make unroll decision
  UnrollDecision decision = makeUnrollDecision(forOp, depAnalysis, memAnalysis);

  llvm::outs() << "  Decision: " 
               << (decision.shouldUnroll ? "UNROLL" : "KEEP") << "\n";
  if (decision.shouldUnroll) {
    llvm::outs() << "  Factor: " << decision.factor << "\n";
    llvm::outs() << "  Unroll-and-Jam: " 
                 << (decision.useUnrollAndJam ? "Yes" : "No") << "\n";
  }
  llvm::outs() << "  Reason: " << decision.reason << "\n";

  // Collect nested loops BEFORE unrolling (to avoid iterator invalidation)
  SmallVector<affine::AffineForOp, 4> nestedLoops;
  if (isValidOperation(forOp)) {
    for (Operation &nestedOp : forOp.getBody()->getOperations()) {
      if (auto nestedForOp = dyn_cast<affine::AffineForOp>(nestedOp)) {
        nestedLoops.push_back(nestedForOp);
      }
    }
  }

  // Perform unroll - AFTER collecting nested loops
  if (decision.shouldUnroll) {
    bool success = false;
    if (decision.useUnrollAndJam) {
      auto outerFor = forOp->getParentOfType<affine::AffineForOp>();
      if (!outerFor) {
        forOp.emitWarning("Cannot jam without outer loop");
        success = performUnroll(forOp, decision.factor);
      } else {
        success = performUnrollAndJam(forOp, decision.factor);
      }
    } else {
      success = performUnroll(forOp, decision.factor);
    }

    if (!success) {
      forOp.emitWarning("Unrolling failed");
    } else {
      llvm::outs() << "  Successfully unrolled!\n";
    }
    
    // After unrolling, forOp is INVALID - don't use it anymore!
    // Return immediately to avoid accessing erased operation
    return;
  }

  // Process nested loops only if we didn't unroll (forOp still valid)
  for (auto nestedLoop : nestedLoops) {
    if (isValidOperation(nestedLoop)) {
      analyzeLoop(nestedLoop);
    }
  }
}

void DependencyAnalysisTestPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Collect all top-level loops first
  SmallVector<affine::AffineForOp, 4> topLevelLoops;
  for (Block &block : func.getBody()) {
    for (Operation &op : block.getOperations()) {
      if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
        topLevelLoops.push_back(forOp);
      }
    }
  }

  // Process each top-level loop
  for (auto forOp : topLevelLoops) {
    if (isValidOperation(forOp)) {
      analyzeLoop(forOp);
    }
  }
}

bool DependencyAnalysisTestPass::performUnroll(
    affine::AffineForOp forOp, unsigned factor) {
  return succeeded(affine::loopUnrollByFactor(forOp, factor));
}

bool DependencyAnalysisTestPass::performUnrollAndJam(
    affine::AffineForOp forOp, unsigned factor) {
  return succeeded(affine::loopUnrollJamByFactor(forOp, factor));
}

void registerDependencyAnalysisTestPass() {
  mlir::PassRegistration<DependencyAnalysisTestPass>();
}

} // namespace mlir