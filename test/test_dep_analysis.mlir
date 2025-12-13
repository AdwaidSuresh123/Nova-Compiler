func.func @test_loop(%arg0: memref<100xf32>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c1 = arith.constant 1 : index
  affine.for %i = 0 to 100 {
    %0 = affine.load %arg0[%i] : memref<100xf32>
    %1 = arith.addf %0, %0 : f32
    affine.store %1, %arg0[%i] : memref<100xf32>
  }
  return
}
