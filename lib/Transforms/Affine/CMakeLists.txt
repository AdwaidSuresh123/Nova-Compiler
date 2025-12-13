#===- CMakeLists.txt - Affine Transform Passes ------------------*- CMake -*-===#

add_mlir_library(AffineTransforms
  DependencyAnalysis.cpp
  DependencyAnalysisTestPass.cpp
  MemoryAccessAnalysis.cpp

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
)
