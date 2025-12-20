# CMakeLists.txt for Nova to tosa Lowering Pass

add_mlir_library(MLIRNovaToTosaTransforms
  NovaToTosa.cpp
  TNovaToTosa.cpp
  NovaToTosa.h

  DEPENDS
  MLIRTosaDialect

  LINK_LIBS PUBLIC
  MLIRTosaDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRNova
)

target_include_directories(MLIRNovaToTosaTransforms PRIVATE
  ${PROJECT_SOURCE_DIR}/include/Compiler/Translation/NovaToTosa
)