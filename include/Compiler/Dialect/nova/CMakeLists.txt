set(LLVM_TARGET_DEFINITIONS NovaOps.td)
mlir_tablegen(NovaOps.h.inc -gen-op-decls)
mlir_tablegen(NovaOps.cpp.inc -gen-op-defs)
mlir_tablegen(NovaOpsDialect.h.inc -gen-dialect-decls -dialect=nova)
mlir_tablegen(NovaOpsDialect.cpp.inc -gen-dialect-defs -dialect=nova)
mlir_tablegen(NovaOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(NovaOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(NovaOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(NovaOpsAttributes.cpp.inc -gen-attrdef-defs)
set(LLVM_TARGET_DEFINITIONS NovaCanonicalizations.td)
mlir_tablegen(NovaCanonicalization.inc -gen-rewriters)
set(LLVM_TARGET_DEFINITIONS NovaOps.td)
mlir_tablegen(NovaPatterns.inc -gen-rewriters)
add_public_tablegen_target(MLIRNovaOpsIncGen)
add_dependencies(mlir-headers MLIRNovaOpsIncGen)




# Install generated header files
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/NovaOps.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/NovaOpsDialect.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/NovaOpsEnums.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/NovaOpsAttributes.h.inc
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Compiler/Dialect/nova"
)