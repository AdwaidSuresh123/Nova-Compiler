#ifndef LIB_TRANSFORM_AFFINE_AFFINEFULLUNROLL_H_
#define LIB_TRANSFORM_AFFINE_AFFINEFULLUNROLL_H_

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"


namespace mlir {
namespace nova {    

#define GEN_PASS_DECL_AFFINEFULLUNROLL
#include "Compiler/Transforms/Passes.h.inc"

}
}

#endif 