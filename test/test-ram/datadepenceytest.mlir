module {
  func.func @test_unroll(%A: memref<32xf32>) {
    affine.for %i = 0 to 32 {
      %val1 = affine.load %A[%i] : memref<32xf32>
      %val2 = affine.load %A[%i + 1] : memref<32xf32>
      %sum = arith.addf %val1, %val2 : f32
      affine.store %sum, %A[%i] : memref<32xf32>
    }
    return
  }
}
