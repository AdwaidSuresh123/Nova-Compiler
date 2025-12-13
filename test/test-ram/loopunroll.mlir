func.func @matmul(%A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>) {
  // Initialize output matrix C to zero
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      %zero = arith.constant 0.0 : f32
      affine.store %zero, %C[%i, %j] : memref<64x64xf32>
    }
  }

  // Matrix multiplication: C = A * B
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 64 {
        %a = affine.load %A[%i, %k] : memref<64x64xf32>
        %b = affine.load %B[%k, %j] : memref<64x64xf32>
        %c = affine.load %C[%i, %j] : memref<64x64xf32>
        %prod = arith.mulf %a, %b : f32
        %sum = arith.addf %c, %prod : f32
        affine.store %sum, %C[%i, %j] : memref<64x64xf32>
      }
    }
  }
  return
}

func.func @main() {
  // Allocate matrices A, B, C
  %A = memref.alloc() : memref<64x64xf32>
  %B = memref.alloc() : memref<64x64xf32>
  %C = memref.alloc() : memref<64x64xf32>

  // Initialize matrices A and B with some values
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      %val = arith.constant 1.0 : f32
      affine.store %val, %A[%i, %j] : memref<64x64xf32>
      affine.store %val, %B[%i, %j] : memref<64x64xf32>
    }
  }

  // Call matmul function
  call @matmul(%A, %B, %C) : (memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>) -> ()

  // Deallocate matrices
  memref.dealloc %A : memref<64x64xf32>
  memref.dealloc %B : memref<64x64xf32>
  memref.dealloc %C : memref<64x64xf32>

  return
}