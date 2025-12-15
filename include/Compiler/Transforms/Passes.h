#ifndef LIB_TRANSFORM_AFFINE_PASSES_H_
#define LIB_TRANSFORM_AFFINE_PASSES_H_

#include "Compiler/Transforms/Affine/AffineFullUnroll.h"
#include "Compiler/Transforms/ParallelizeOuterLoops.h"
#include "Compiler/Transforms/FuseMatmulBias.h"

namespace mlir {
namespace nova {

#define GEN_PASS_REGISTRATION
#include "Compiler/Transforms/Passes.h.inc"

}  // namespace nova
}  // namespace mlir

#endif