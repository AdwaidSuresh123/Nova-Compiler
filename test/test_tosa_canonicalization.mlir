// RUN: nova-opt --canonicalize %s | FileCheck %s



// CHECK-LABEL: func @test_concat_optimization
func.func @test_concat_optimization(%arg0: tensor<1x10x10x1xf32>) -> tensor<1x10x10x1xf32> {
  %0 = tosa.concat %arg0 {axis = 0 : i32} : (tensor<1x10x10x1xf32>) -> tensor<1x10x10x1xf32>
  return %0 : tensor<1x10x10x1xf32>
}

// CHECK-LABEL: func @test_select_canonicalize
func.func @test_select_canonicalize(%arg0: tensor<1x10x10x1xi1>, %arg1: tensor<1x10x10x1xf32>, %arg2: tensor<1x10x10x1xf32>) -> tensor<1x10x10x1xf32> {
  %0 = tosa.logical_not %arg0 : (tensor<1x10x10x1xi1>) -> tensor<1x10x10x1xi1>
  %1 = tosa.select %0, %arg1, %arg2 : (tensor<1x10x10x1xi1>, tensor<1x10x10x1xf32>, tensor<1x10x10x1xf32>) -> tensor<1x10x10x1xf32>
  return %1 : tensor<1x10x10x1xf32>
}

// CHECK-LABEL: func @test_consolidate_transpose
func.func @test_consolidate_transpose(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x4x3x2xf32> {
  %0 = tosa.transpose %arg0 {perms = array<i32: 1, 2, 3, 0>} : (tensor<1x2x3x4xf32>) -> tensor<2x3x4x1xf32>
  %1 = tosa.transpose %0 {perms = array<i32: 3, 2, 1, 0>} : (tensor<2x3x4x1xf32>) -> tensor<1x4x3x2xf32>
  return %1 : tensor<1x4x3x2xf32>
}

// CHECK-LABEL: func @test_transpose_is_reshape
func.func @test_transpose_is_reshape(%arg0: tensor<1x10x10x1xf32>) -> tensor<1x1x10x10xf32> {
  %0 = tosa.transpose %arg0 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x10x10x1xf32>) -> tensor<1x1x10x10xf32>
  return %0 : tensor<1x1x10x10xf32>
}
// CHECK-LABEL: func @test_clamp_is_noop
func.func @test_clamp_is_noop(%arg0: tensor<10xi8>) -> tensor<10xi8> {
  // CHECK-NOT: tosa.clamp
  // CHECK: return %arg0
  %0 = tosa.clamp %arg0 {min_val = -128 : i8, max_val = 127 : i8} : (tensor<10xi8>) -> tensor<10xi8>
  return %0 : tensor<10xi8>
}

// CHECK-LABEL: func @test_clamp_clamp_optimization
func.func @test_clamp_clamp_optimization(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.clamp
  // CHECK-SAME: min_val = 2.000000e-01 : f32
  // CHECK-SAME: max_val = 8.000000e-01 : f32
  // CHECK-NOT: tosa.clamp
  %0 = tosa.clamp %arg0 {min_val = 0.0 : f32, max_val = 1.0 : f32} : (tensor<10xf32>) -> tensor<10xf32>
  %1 = tosa.clamp %0 {min_val = 0.2 : f32, max_val = 0.8 : f32} : (tensor<10xf32>) -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK-LABEL: func @test_concat_slice_optimization
func.func @test_concat_slice_optimization(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<2xf32> {
  // CHECK: tosa.slice %arg0
  // CHECK-NOT: tosa.concat
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<4xf32>, tensor<4xf32>) -> tensor<8xf32>
  %1 = "tosa.const_shape"() {values = dense<[1]> : tensor<1xindex>} : () -> !tosa.shape<1>
  %2 = "tosa.const_shape"() {values = dense<[2]> : tensor<1xindex>} : () -> !tosa.shape<1>
  %3 = tosa.slice %0, %1, %2 : (tensor<8xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<2xf32>
  return %3 : tensor<2xf32>
}

// CHECK-LABEL: func @test_pad_slice_optimization
func.func @test_pad_slice_optimization(%arg0: tensor<4x4xf32>) -> tensor<2x2xf32> {
  // CHECK: tosa.slice %arg0
  // CHECK-NOT: tosa.pad
  %0 = "tosa.const_shape"() {values = dense<[1, 1, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %1 = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %2 = tosa.pad %arg0, %0, %1 : (tensor<4x4xf32>, !tosa.shape<4>, tensor<1xf32>) -> tensor<6x6xf32>
  %3 = "tosa.const_shape"() {values = dense<[2, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %4 = "tosa.const_shape"() {values = dense<[2, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %5 = tosa.slice %2, %3, %4 : (tensor<6x6xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x2xf32>
  return %5 : tensor<2x2xf32>
}

// CHECK-LABEL: func @test_slice_dynamic_size
func.func @test_slice_dynamic_size(%arg0: tensor<10x10xf32>) -> tensor<5x5xf32> {
  // CHECK: tosa.slice
  // CHECK-SAME: size = [5, 5]
  %0 = "tosa.const_shape"() {values = dense<[0, 0]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = "tosa.const_shape"() {values = dense<[-1, -1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %2 = tosa.slice %arg0, %0, %1 : (tensor<10x10xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<5x5xf32>
  return %2 : tensor<5x5xf32>
}

// CHECK-LABEL: func @test_fold_add_zero
func.func @test_fold_add_zero(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: return %arg0
  %0 = "tosa.const"() {values = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32>
  %1 = tosa.add %arg0, %0 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func @test_fold_mul_one
func.func @test_fold_mul_one(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: return %arg0
  %0 = "tosa.const"() {values = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32>
  %1 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %2 = tosa.mul %arg0, %0, %1 : (tensor<4xf32>, tensor<4xf32>, tensor<1xi8>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

// CHECK-LABEL: func @test_fold_pad_zero
func.func @test_fold_pad_zero(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: return %arg0
  %0 = "tosa.const_shape"() {values = dense<[0, 0]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %2 = tosa.pad %arg0, %0, %1 : (tensor<10xf32>, !tosa.shape<2>, tensor<1xf32>) -> tensor<10xf32>
  return %2 : tensor<10xf32>
}

// CHECK-LABEL: func @test_fold_transpose_identity
func.func @test_fold_transpose_identity(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
  // CHECK: return %arg0
  %0 = tosa.transpose %arg0 {perms = array<i32: 0, 1, 2, 3>} : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>
}

// CHECK-LABEL: func @test_fold_abs_abs
func.func @test_fold_abs_abs(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.abs
  // CHECK-NOT: tosa.abs
  %0 = tosa.abs %arg0 : (tensor<10xf32>) -> tensor<10xf32>
  %1 = tosa.abs %0 : (tensor<10xf32>) -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK-LABEL: func @test_fold_negate_negate
func.func @test_fold_negate_negate(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: return %arg0
  %zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %0 = tosa.negate %arg0, %zp, %zp : (tensor<10xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<10xf32>
  %1 = tosa.negate %0, %zp, %zp : (tensor<10xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK-LABEL: func @test_fold_slice_identity
func.func @test_fold_slice_identity(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: return %arg0
  %0 = "tosa.const_shape"() {values = dense<[0]> : tensor<1xindex>} : () -> !tosa.shape<1>
  %1 = "tosa.const_shape"() {values = dense<[10]> : tensor<1xindex>} : () -> !tosa.shape<1>
  %2 = tosa.slice %arg0, %0, %1 : (tensor<10xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<10xf32>
  return %2 : tensor<10xf32>
}

// CHECK-LABEL: func @test_fold_tile_identity
func.func @test_fold_tile_identity(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: return %arg0
  %0 = "tosa.const_shape"() {values = dense<[1]> : tensor<1xindex>} : () -> !tosa.shape<1>
  %1 = tosa.tile %arg0, %0 : (tensor<10xf32>, !tosa.shape<1>) -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK-LABEL: func @test_fold_resize_identity
func.func @test_fold_resize_identity(%arg0: tensor<1x10x10x1xf32>) -> tensor<1x10x10x1xf32> {
  // CHECK: return %arg0
  %scale = "tosa.const_shape"() {values = dense<[1, 1, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %offset = "tosa.const_shape"() {values = dense<[0, 0]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %border = "tosa.const_shape"() {values = dense<[0, 0]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %0 = tosa.resize %arg0, %scale, %offset, %border {mode = "NEAREST_NEIGHBOR"} : (tensor<1x10x10x1xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x10x10x1xf32>
  return %0 : tensor<1x10x10x1xf32>
}
