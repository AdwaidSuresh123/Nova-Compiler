//module {
//  func.func @main(%arg0: tensor<1x2x3xcomplex<f32>>) -> tensor<1x2x3xcomplex<f32>> {
//    %1=nova.sin %arg0  : tensor<1x2x3xcomplex<f32>>
// return %1 :tensor<1x2x3xcomplex<f32>>
// }
//}
module {
  func.func @main(%arg0: tensor<2x2x2xi32>) -> tensor<2x2x2xf32> {
    %1=nova.softmax %arg0  : tensor<2x2x2xi32>
 return %1 :tensor<2x2x2xf32>
 }
}
//module {
//  func.func @main(%arg0: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
//    %1=nova.sin %arg0  : tensor<1x2x3xf32>
// return %1 :tensor<1x2x3xf32>
// }
//}
//../build/tools/nova-opt/nova-opt unary.mlir   --pass-pipeline='builtin.module(canonicalize,func.func(
 //convert-nova-to-linalg,tosa-to-linalg-named,tosa-to-linalg))'
 //1x2x3xcomplex<f32>
