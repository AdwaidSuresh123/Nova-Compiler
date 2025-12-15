# Library subdirectories will go here
add_subdirectory(Dialect)
add_subdirectory(Transforms)
add_subdirectory(Pipeline)
add_subdirectory(API)

# --- Additions for Export/Installation ---

# 1. Install all libraries
install(
    TARGETS 
        MLIRNova
        CompilerTransforms
        NovaPipelines
        MLIRAffineTransformPasses
        AffineTransforms
        MLIRNovaToArithTransforms
        MLIRNovaToLinalgTransforms
        MLIRNovaToTosaTransforms
        NovaCompilerAPI
    EXPORT mlir-compilerTargets  # Name the export set
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

# 2. Export the targets for use by other projects (build tree)
# This creates the 'mlir-compilerTargets.cmake' file in the build directory
export(
    EXPORT mlir-compilerTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/mlir-compilerTargets.cmake"
    NAMESPACE MlirCompiler::
)

# 3. Install the targets for use by other projects (install tree)
# This creates the 'mlir-compilerTargets.cmake' file in the installation directory
install(
    EXPORT mlir-compilerTargets
    FILE mlir-compilerTargets.cmake
    NAMESPACE MlirCompiler::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mlir-compiler"
)