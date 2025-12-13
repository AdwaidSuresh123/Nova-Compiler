// Define affine map for subtraction at top level
#map_sub = affine_map<(d0) -> (d0 - 2)>

func.func @complex_test(%A: memref<32x32xf32>, %B: memref<32x32xf32>, %C: memref<32x32xf32>, %D: memref<32x32xf32>,%arg0: memref<64xf32>) {
  // Extra loop with cross-element dependencies
  affine.for %i = 2 to 31 {
    affine.for %j = 2 to 31 {
      %i_minus_1 = affine.apply #map_sub(%i)
      %j_minus_1 = affine.apply #map_sub(%j)
      %c1 = affine.load %C[%i, %j] : memref<32x32xf32>
      %c2 = affine.load %C[%i_minus_1, %j] : memref<32x32xf32>
      %c3 = affine.load %C[%i, %j_minus_1] : memref<32x32xf32>
      %sum1 = arith.addf %c1, %c2 : f32
      %sum2 = arith.addf %sum1, %c3 : f32
      affine.store %sum2, %D[%i, %j] : memref<32x32xf32>
    }
  }
  
  // Some diagonal write-after-read dependencies
   affine.for %i = 1 to 10 {
      %cst2 = arith.constant 2.000000e+00 : f32  
      affine.store %cst2, %arg0[%i] : memref<64xf32>
    }
  
  return
}