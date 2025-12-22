// RUN: nova-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: func @test_erase_self_copy
func.func @test_erase_self_copy(%arg0: memref<10xf32>) {
  linalg.copy ins(%arg0 : memref<10xf32>) outs(%arg0 : memref<10xf32>)
  return
}

// CHECK-LABEL: func @test_fold_fill_reshape
func.func @test_fold_fill_reshape(%arg0: tensor<10x10xf32>) -> tensor<100xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<10x10xf32> into tensor<100xf32>
  return %1 : tensor<100xf32>
}

// CHECK-LABEL: func @test_fold_fill_pad
func.func @test_fold_fill_pad(%arg0: tensor<10x10xf32>) -> tensor<12x12xf32> {
  // CHECK-NOT: tensor.pad
  // CHECK: linalg.fill
  // CHECK-SAME: outs(%{{.*}} : tensor<12x12xf32>)
  %cst = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = tensor.pad %0 low[1, 1] high[1, 1] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %cst : f32
  } : tensor<10x10xf32> to tensor<12x12xf32>
  return %1 : tensor<12x12xf32>
}

// CHECK-LABEL: func @test_fold_fill_extract
func.func @test_fold_fill_extract(%arg0: tensor<10x10xf32>) -> f32 {
  // CHECK-NOT: tensor.extract
  // CHECK: return %cst
  %cst = arith.constant 1.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32>
  %c0 = arith.constant 0 : index
  %1 = tensor.extract %0[%c0, %c0] : tensor<10x10xf32>
  return %1 : f32
}

// CHECK-LABEL: func @test_fold_fill_transpose
func.func @test_fold_fill_transpose(%arg0: tensor<10x20xf32>, %arg1: tensor<20x10xf32>) -> tensor<20x10xf32> {
  // CHECK-NOT: linalg.transpose
  // CHECK: linalg.fill
  // CHECK-SAME: outs(%arg1 : tensor<20x10xf32>)
  %cst = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<10x20xf32>) -> tensor<10x20xf32>
  %1 = linalg.transpose ins(%0 : tensor<10x20xf32>) outs(%arg1 : tensor<20x10xf32>) permutation = [1, 0]
  return %1 : tensor<20x10xf32>
}

// CHECK-LABEL: func @test_fold_concat_fill
func.func @test_fold_concat_fill(%arg0: tensor<10xf32>, %arg1: tensor<20xf32>) -> tensor<30xf32> {
  // CHECK-NOT: tensor.concat
  // CHECK: linalg.fill
  // CHECK-SAME: outs(%{{.*}} : tensor<30xf32>)
  %cst = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<10xf32>) -> tensor<10xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<20xf32>) -> tensor<20xf32>
  %2 = tensor.concat dim(0) %0, %1 : (tensor<10xf32>, tensor<20xf32>) -> tensor<30xf32>
  return %2 : tensor<30xf32>
}

// CHECK-LABEL: func @test_erase_identity_generic
func.func @test_erase_identity_generic(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<10xf32>) outs(%arg0 : tensor<10xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: func @test_fold_transpose_transpose
func.func @test_fold_transpose_transpose(%arg0: tensor<10x20x30xf32>, %arg1: tensor<30x20x10xf32>) -> tensor<30x20x10xf32> {
  // CHECK-NOT: linalg.transpose
  // CHECK: linalg.transpose
  // CHECK-SAME: permutation = [2, 1, 0]
  %0 = tensor.empty() : tensor<20x30x10xf32>
  %1 = linalg.transpose ins(%arg0 : tensor<10x20x30xf32>) outs(%0 : tensor<20x30x10xf32>) permutation = [1, 2, 0]
  %2 = linalg.transpose ins(%1 : tensor<20x30x10xf32>) outs(%arg1 : tensor<30x20x10xf32>) permutation = [1, 0, 2]
  return %2 : tensor<30x20x10xf32>
}

// CHECK-LABEL: func @test_erase_identity_broadcast
func.func @test_erase_identity_broadcast(%arg0: tensor<10x20xf32>) -> tensor<10x20xf32> {
  // CHECK-NOT: linalg.broadcast
  // CHECK: return %arg0
  %0 = linalg.broadcast ins(%arg0 : tensor<10x20xf32>) outs(%arg0 : tensor<10x20xf32>) dimensions = []
  return %0 : tensor<10x20xf32>
}

// CHECK-LABEL: func @test_erase_dead_linalg_op
func.func @test_erase_dead_linalg_op(%arg0: memref<0xf32>, %arg1: memref<0xf32>) {
  // CHECK-NOT: linalg.generic
  linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : memref<0xf32>) outs(%arg1 : memref<0xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  }
  return
}

// CHECK-LABEL: func @test_fold_tensor_cast_consumer
func.func @test_fold_tensor_cast_consumer(%arg0: tensor<?x?xf32>, %arg1: tensor<10x20xf32>) -> tensor<10x20xf32> {
  // CHECK-NOT: tensor.cast
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}} : tensor<10x20xf32>)
  %0 = tensor.cast %arg0 : tensor<?x?xf32> to tensor<10x20xf32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%0 : tensor<10x20xf32>) outs(%arg1 : tensor<10x20xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<10x20xf32>
  return %1 : tensor<10x20xf32>
}

// CHECK-LABEL: func @test_fold_fill_copy
func.func @test_fold_fill_copy(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK-NOT: linalg.copy
  // CHECK: linalg.fill
  %cst = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = tensor.empty() : tensor<10x10xf32>
  %2 = linalg.copy ins(%0 : tensor<10x10xf32>) outs(%1 : tensor<10x10xf32>) -> tensor<10x10xf32>
  return %2 : tensor<10x10xf32>
}

// CHECK-LABEL: func @test_fold_fill_copy_overwrite
func.func @test_fold_fill_copy_overwrite(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK-NOT: linalg.fill
  // CHECK: linalg.copy
  // CHECK-SAME: outs(%arg0 : tensor<10x10xf32>)
  %cst = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = linalg.copy ins(%arg1 : tensor<10x10xf32>) outs(%0 : tensor<10x10xf32>) -> tensor<10x10xf32>
  return %1 : tensor<10x10xf32>
}

// CHECK-LABEL: func @test_fold_insert_pad_into_fill
func.func @test_fold_insert_pad_into_fill(%arg0: tensor<10x10xf32>) -> tensor<20x20xf32> {
  // CHECK-NOT: tensor.pad
  // CHECK: linalg.fill
  // CHECK: tensor.insert_slice
  %cst = arith.constant 0.0 : f32
  %pad_val = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = tensor.pad %0 low[2, 2] high[3, 3] {
  ^bb0(%arg2: index, %arg3: index):
    tensor.yield %pad_val : f32
  } : tensor<10x10xf32> to tensor<15x15xf32>
  %2 = tensor.empty() : tensor<20x20xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<20x20xf32>) -> tensor<20x20xf32>
  %4 = tensor.insert_slice %1 into %3[0, 0] [15, 15] [1, 1] : tensor<15x15xf32> into tensor<20x20xf32>
  return %4 : tensor<20x20xf32>
}

// CHECK-LABEL: func @test_unpack_pack_cancellation
func.func @test_unpack_pack_cancellation(%arg0: tensor<128x256xf32>, %arg1: tensor<16x8x8x32xf32>) -> tensor<128x256xf32> {
  // CHECK-NOT: linalg.pack
  // CHECK-NOT: linalg.unpack
  // CHECK: return %arg0
  %0 = linalg.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %arg1 : tensor<128x256xf32> -> tensor<16x8x8x32xf32>
  %1 = linalg.unpack %0 inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %arg0 : tensor<16x8x8x32xf32> -> tensor<128x256xf32>
  return %1 : tensor<128x256xf32>
}

// CHECK-LABEL: func @test_fold_tensor_cast_pack
func.func @test_fold_tensor_cast_pack(%arg0: tensor<128x256xf32>, %arg1: tensor<16x8x8x32xf32>) -> tensor<16x8x8x32xf32> {
  // CHECK-NOT: tensor.cast
  // CHECK: linalg.pack
  %0 = tensor.cast %arg0 : tensor<128x256xf32> to tensor<?x?xf32>
  %1 = linalg.pack %0 inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %arg1 : tensor<?x?xf32> -> tensor<16x8x8x32xf32>
  return %1 : tensor<16x8x8x32xf32>
}

// CHECK-LABEL: func @test_fold_tensor_cast_unpack
func.func @test_fold_tensor_cast_unpack(%arg0: tensor<16x8x8x32xf32>, %arg1: tensor<128x256xf32>) -> tensor<128x256xf32> {
  // CHECK-NOT: tensor.cast
  // CHECK: linalg.unpack
  %0 = tensor.cast %arg0 : tensor<16x8x8x32xf32> to tensor<?x?x8x32xf32>
  %1 = linalg.unpack %0 inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %arg1 : tensor<?x?x8x32xf32> -> tensor<128x256xf32>
  return %1 : tensor<128x256xf32>
}
