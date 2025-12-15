add_llvm_executable(nova-opt nova-opt.cpp)

llvm_update_compile_flags(nova-opt)
target_link_libraries(nova-opt PRIVATE
  MLIROptLib
  MLIRMlirOptMain
  MLIRSupport
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRAnalysis
  MLIRTransformUtils
  MLIRRewrite
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRDialect
  MLIRDialectUtils

  # Your custom libraries
  MLIRNova
  CompilerTransforms
  MLIRAffineTransformPasses
  
  MLIRNovaToArithTransforms
  MLIRNovaToLinalgTransforms
  MLIRNovaToTosaTransforms
  NovaPipelines

  # Core dialects that we know exist and are needed
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRArithDialect
  MLIRArithTransforms
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgTransformOps
  MLIRLinalgUtils
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMathTransforms
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRVectorDialect
  MLIRVectorTransforms

  # LLVM IR translation support
  MLIRToLLVMIRTranslationRegistration
  MLIRLLVMToLLVMIRTranslation


)

install(TARGETS nova-opt RUNTIME DESTINATION bin)
