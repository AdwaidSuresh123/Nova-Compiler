#===- CMakeLists.txt - Affine Transform Passes ------------------*- CMake -*-===#

add_mlir_dialect_library(MLIRAffineTransformPasses
  AffineFullUnroll.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/Affine/DependencyAnalysis.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/Affine/DependencyAnalysisTestPass.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/Affine/MemoryAccessAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Compiler/Transforms/Affine

  LINK_LIBS PUBLIC
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRAffineAnalysis
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRArithDialect
    MLIRPass
    MLIRTransforms
    MLIRIR
    MLIRSupport
    MLIRTransformUtils
    MLIRAnalysis
)

target_include_directories(MLIRAffineTransformPasses PRIVATE
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include/Compiler/Transforms/Affine
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

install(FILES
  ${PROJECT_SOURCE_DIR}/include/Compiler/Transforms/Affine/AffineFullUnroll.h
  ${PROJECT_SOURCE_DIR}/include/Compiler/Transforms/Affine/DependencyAnalysis.h
  ${PROJECT_SOURCE_DIR}/include/Compiler/Transforms/Affine/DependencyAnalysisTestPass.h
  ${PROJECT_SOURCE_DIR}/include/Compiler/Transforms/Affine/MemoryAccessAnalysis.h
  DESTINATION include/Compiler/Transforms/Affine
)
if(MLIR_INCLUDE_TESTS)
  add_subdirectory(test)
endif() 