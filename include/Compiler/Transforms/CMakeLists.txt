# Transform definitions will go here
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Affine)
add_public_tablegen_target(NovaAffineFullUnrollPassIncGen)

#  pass documentation
add_mlir_doc(AffineFullUnroll AffineFullUnrollPass Passes/ -gen-pass-doc)

# Building the pass library
add_mlir_library(AffineFullUnroll
  AffineFullUnroll.cpp
  AffineFullUnroll.h
  
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Compiler/Transforms
  
  DEPENDS
  NovaAffineFullUnrollPassIncGen
  
  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRAffineAnalysis
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
)

# include directories
target_include_directories(AffineFullUnroll PRIVATE
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Install generated header files
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/Passes.h.inc
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Compiler/Transforms"
)

add_subdirectory(Affine)