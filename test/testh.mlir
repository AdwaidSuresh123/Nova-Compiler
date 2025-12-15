func.func @main1(%arg0: tensor<2x2xcomplex<i32>>,%arg1:tensor<2x2xcomplex<i32>> ) -> tensor<2x2xcomplex<i32>> {
  %2 = nova.max %arg1,%arg1 : tensor<2x2xcomplex<i32>>,tensor<2x2xcomplex<i32>>
  return %2 :tensor<2x2xcomplex<i32>>
  }
