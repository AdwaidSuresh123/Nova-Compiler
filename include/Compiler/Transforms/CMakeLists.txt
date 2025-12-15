# Transform definitions will go here
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Affine)
add_public_tablegen_target(NovaAffineFullUnrollPassIncGen)

# Install generated header files
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/Passes.h.inc
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Compiler/Transforms"
)

add_subdirectory(Affine)