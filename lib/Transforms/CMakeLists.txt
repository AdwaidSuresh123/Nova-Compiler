# Add CleanupPass to the list of sources
add_mlir_library(CompilerTransforms
  CleanupPass.cpp
  FuseMatmulBias.cpp
  FastmathFlag.cpp
  ParallelizeOuterLoops.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLLVMDialect
  MLIRTransformUtils
  MLIRNova
  MLIRArithDialect
  MLIRMathDialect
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRLinalgDialect
)
add_subdirectory(Affine)