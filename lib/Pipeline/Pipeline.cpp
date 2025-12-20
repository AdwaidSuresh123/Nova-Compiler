//convert to llvm includes
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
//other conversion includes from mlir
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
//utils
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
//buffer includes
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"

//nova dialect includes
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"

//optimization passes includes
#include "Compiler/Transforms/CleanupPass.h"
#include "Compiler/Transforms/Affine/AffineFullUnroll.h"
#include "Compiler/Transforms/FastmathFlag.h"
#include "Compiler/Transforms/ParallelizeOuterLoops.h"
#include "Compiler/Transforms/FuseMatmulBias.h"

//lowering passes
#include "Compiler/Translation/NovaToArith/NovaToArith.h"
#include "Compiler/Translation/NovaToTosa/NovaToTosa.h"
#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"

// header of this file
#include "Compiler/Pipeline/Pipeline.h"
#include "Compiler/Transforms/Affine/DependencyAnalysisTestPass.h"

using namespace mlir;

namespace mlir {
namespace nova {
#define GEN_PASS_REGISTRATION
#include "Compiler/Transforms/Passes.h.inc"
}} // namespace mlir::nova

void mlir::nova::createNovaPipelines(OpPassManager &pm) {
  //pm.addPass(createCanonicalizerPass());
  pm.addPass(std::make_unique<DependencyAnalysisTestPass>());
  
  // Lower Nova dialect to standard dialects 
//pm.addNestedPass<func::FuncOp>(ceateNovaToLinalg());  
  pm.addPass(createCanonicalizerPass()); 
  pm.addPass(createNovaToTosaLoweringPass());
  pm.addNestedPass<func::FuncOp>(createNovaToLinalgLoweringPass());
  pm.addPass(createNovaToArithLoweringPass());
  pm.addPass(createCanonicalizerPass());
//  pm.addPass(mlir::createTensorBufferizePass());  
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

//tosa to linalg named ops
pm.addNestedPass<func::FuncOp>(mlir::tosa::createTosaToLinalgNamed()); 
  //tosa to linalg 
pm.addNestedPass<func::FuncOp>(mlir::tosa::createTosaToLinalg());  
//tosa to tensors
pm.addPass(mlir::createTosaToTensorPass());

  // Convert elementwise operations to Linalg
  pm.addPass(mlir::createConvertElementwiseToLinalgPass());
  
  //  Bufferization (Tensor -> MemRef)
  bufferization::OneShotBufferizePassOptions bufferizeOptions;  // create a options variable
  bufferizeOptions.bufferizeFunctionBoundaries = true;        // mention that we want to bufferize function boundaries ie parameters and returns
  bufferizeOptions.functionBoundaryTypeConversion=            //keep the layout map as identity for function boundaries
                bufferization::LayoutMapOption::IdentityLayoutMap;      // this will ensure that the function signatures remain unchanged

  // so when we bufferize a function that takes a tensor as input, it will be converted to a memref with an identity layout map
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizeOptions));
  
  //  Buffer deallocation pipeline - this will make sure that all allocated buffers are deallocated properly based on the analysis done during bufferization
  bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  bufferization::buildBufferDeallocationPipeline(pm, deallocationOptions);
  
  //Convert remaining bufferization ops to memref
  pm.addPass(mlir::createConvertBufferizationToMemRefPass());
  

  // Lower Linalg to loops (SCF dialect)
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  
  // Convert SCF to CF (Control Flow)
  pm.addPass(mlir::createSCFToControlFlowPass());
  //pm.addPass(createCanonicalizerPass()); 
  
  //Lower to LLVM dialect
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass()); // Convert functions lastly
  
  pm.addPass(mlir::compiler::createCleanupPass());
  // reconcile unrealized casts
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  
  // Final canonicalization
  //pm.addPass(createCanonicalizerPass());
}

void mlir::nova::registerNovaPipelines() {
  PassPipelineRegistration<>("nova-opt-pipeline",
                            "Nova Optimizer Pipeline",
                            createNovaPipelines);
}