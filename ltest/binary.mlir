module {
  func.func @main(%arg0: tensor<1x8x3xf32>,%arg1:tensor<1x8x3xi16>) -> tensor<1x8x3xf32> {
    %1=nova.div %arg0, %arg1 : tensor<1x8x3xf32> ,tensor<1x8x3xi16>
 return %1 :tensor<1x8x3xf32>
 }
}
// ../build/tools/nova-opt/nova-opt binary.mlir   --pass-pipeline='builtin.module(canonicalize,func.func(
// convert-nova-to-linalg,tosa-to-linalg-named,tosa-to-linalg))'
