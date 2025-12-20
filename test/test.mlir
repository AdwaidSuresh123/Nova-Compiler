module {
 // func.func @main(%arg0: tensor<2x2xf32>,%arg1:tensor<2x2xf32>,%arg2:tensor<2x2xf32> ) -> tensor<2x2xf32> {
 //   %5 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
 //   %6 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
 //   %7 = tosa.matmul %arg0, %arg1, %5, %6 : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x2xf32> 
    //%2 = nova.add %1,%arg2 : tensor<2x2xf32>,tensor<2x2xf32>
 // return %7 :tensor<2x2xf32>
 // }
  
  func.func @main1(%arg0: tensor<2x2x2xf32>,%arg1:tensor<2x2xf32>,%arg2:tensor<1x2x2xf32> ) -> tensor<2x2x2xf32> {
  %1 =nova.matmul %arg0,%arg1 : tensor<2x2x2xf32>,tensor<2x2xf32>
  %2 = nova.add %1,%arg2 : tensor<2x2x2xf32>,tensor<1x2x2xf32>
  return %2 :tensor<2x2x2xf32>
  }

 // func.func @main2(%arg0: tensor<2x2xf32>,%arg1:tensor<2x2xf32>,%arg2:tensor<3x3xi16> ) -> tensor<2x2xf32> {
 // %1 =nova.matmul %arg0,%arg1 : tensor<2x2xf32>,tensor<2x2xf32>
 // %2 = nova.add %1,%arg2 : tensor<2x2xf32>,tensor<3x3xi16>
 // return %2 :tensor<2x2xf32>
  //}
  
 // func.func @main(%arg0: tensor<2x2xf32>,%arg1:tensor<2x2xf32>,%arg2:tensor<3x3xi64> ) -> tensor<3x3xf64> {
 // %1 =nova.matmul %arg0,%arg1 : tensor<2x2xf32>,tensor<2x2xf32>
 // %2 = nova.add %1,%arg2 : tensor<2x2xf32>,tensor<3x3xi64>
 // return %2 :tensor<3x3xf64>
 // }
  }


