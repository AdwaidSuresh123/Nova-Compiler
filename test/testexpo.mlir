module {
  func.func @test_tensor_fold_chain(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    %c1 = nova.constant {value = dense<[5.0]> : tensor<1xf32>} : tensor<1xf32>
  %c2 = nova.constant {value = dense<[-8.0]> : tensor<1xf32>} : tensor<1xf32>
    %tmp = nova.sub %arg0, %c1 : tensor<1xf32>, tensor<1xf32>
    %res = nova.add %c2, %c2 : tensor<1xf32>, tensor<1xf32> 
    return %res : tensor<1xf32>
  }
}