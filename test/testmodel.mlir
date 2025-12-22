module {
  // Helper Function: Embedding Lookup
  func.func @embedding_lookup(%indices: tensor<8x128xi32>, %W: tensor<1000x768xf32>) -> tensor<8x128x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c128 = arith.constant 128 : index
    %c768 = arith.constant 768 : index
    
    // Initialize output tensor
    %init = tensor.empty() : tensor<8x128x768xf32>
    
    // Use linalg.generic for embedding lookup
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1)>,  // indices
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>  // output
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%indices : tensor<8x128xi32>) outs(%init : tensor<8x128x768xf32>) {
    ^bb0(%idx: i32, %out: f32):
      %idx_cast = arith.index_cast %idx : i32 to index
      %d2 = linalg.index 2 : index
      %val = tensor.extract %W[%idx_cast, %d2] : tensor<1000x768xf32>
      linalg.yield %val : f32
    } -> tensor<8x128x768xf32>
    
    return %output : tensor<8x128x768xf32>
  }

  // Helper Function: Matrix Multiplication with nova
  func.func @matmul(%A: tensor<8x128x768xf32>, %B: tensor<768x768xf32>) -> tensor<8x128x768xf32> {
    // Reshape for matmul: collapse batch dimension
    %A_2d = tensor.collapse_shape %A [[0, 1], [2]] : tensor<8x128x768xf32> into tensor<1024x768xf32>
    
    // nova.matmul doesn't use explicit output buffer in its syntax based on NovaOps.td
    %result_2d = nova.matmul %A_2d, %B : tensor<1024x768xf32>, tensor<768x768xf32>
    
    %result = tensor.expand_shape %result_2d [[0, 1], [2]] output_shape [8, 128, 768] : tensor<1024x768xf32> into tensor<8x128x768xf32>
    return %result : tensor<8x128x768xf32>
  }

  // Helper Function: Multi-Head Self-Attention
  func.func @self_attention(%X: tensor<8x128x768xf32>, %W_Q: tensor<768x768xf32>, %W_K: tensor<768x768xf32>, %W_V: tensor<768x768xf32>, %W_O: tensor<768x768xf32>) -> tensor<8x128x768xf32> {
    
    // Linear projections: Q = X * W_Q, K = X * W_K, V = X * W_V
    %Q = func.call @matmul(%X, %W_Q) : (tensor<8x128x768xf32>, tensor<768x768xf32>) -> tensor<8x128x768xf32>
    %K = func.call @matmul(%X, %W_K) : (tensor<8x128x768xf32>, tensor<768x768xf32>) -> tensor<8x128x768xf32>
    %V = func.call @matmul(%X, %W_V) : (tensor<8x128x768xf32>, tensor<768x768xf32>) -> tensor<8x128x768xf32>
    
    // Compute attention scores: Q * K^T
    // Reshape for batched matmul: (8, 128, 768) x (8, 768, 128) -> (8, 128, 128)
    
    // Transpose K: swap dim 1 and 2.
    %K_transposed = nova.transpose %K axes1=1 axes2=2 : tensor<8x128x768xf32>
    
    // Batched matmul for attention scores
    // Assuming nova.matmul supports batched matmul (broadcasting or implicit batching)
    // Q: [8, 128, 768], K_T: [8, 768, 128] -> [8, 128, 128]
    // Note: The result type of transpose might need to be inferred or explicit. 
    // If nova.transpose returns tensor<8x768x128xf32>, then we use that.
    
    %scores_raw = nova.matmul %Q, %K_transposed : tensor<8x128x768xf32>, tensor<8x768x128xf32>
    
    // Scale scores
    // Create constant tensor for scale
    %scale_tensor = nova.constant {value = dense<27.7128> : tensor<1xf32>} : tensor<1xf32>
    
    // Broadcast division
    %scores = nova.div %scores_raw, %scale_tensor : tensor<8x128x128xf32>, tensor<1xf32>
    
    // Softmax would go here
    // %softmax_scores = nova.softmax %scores dimension=2 : tensor<8x128x128xf32>
    // For now keeping it as is (skipped in original) or adding it if requested. 
    // The user said "what ever u can convert". I'll add nova.softmax.
    %softmax_scores = nova.softmax %scores dimension=2 : tensor<8x128x128xf32>
    
    // Attention: scores * V
    // scores: [8, 128, 128], V: [8, 128, 768] -> [8, 128, 768]
    %attn = nova.matmul %softmax_scores, %V : tensor<8x128x128xf32>, tensor<8x128x768xf32>
    
    // Final projection: attn * W_O
    %output = func.call @matmul(%attn, %W_O) : (tensor<8x128x768xf32>, tensor<768x768xf32>) -> tensor<8x128x768xf32>
    
    return %output : tensor<8x128x768xf32>
  }

  // Helper Function: Feed-Forward Network
  func.func @feed_forward(%X: tensor<8x128x768xf32>, %W1: tensor<768x3072xf32>, %W2: tensor<3072x768xf32>) -> tensor<8x128x768xf32> {
    
    // First linear layer: X * W1
    %X_2d = tensor.collapse_shape %X [[0, 1], [2]] : tensor<8x128x768xf32> into tensor<1024x768xf32>
    
    %hidden_2d = nova.matmul %X_2d, %W1 : tensor<1024x768xf32>, tensor<768x3072xf32>
    
    %hidden = tensor.expand_shape %hidden_2d [[0, 1], [2]] output_shape [8, 128, 3072] : tensor<1024x3072xf32> into tensor<8x128x3072xf32>
    
    // ReLU activation
    %relu = nova.relu %hidden : tensor<8x128x3072xf32>
    
    // Second linear layer: relu * W2
    %relu_2d = tensor.collapse_shape %relu [[0, 1], [2]] : tensor<8x128x3072xf32> into tensor<1024x3072xf32>
    
    %output_2d = nova.matmul %relu_2d, %W2 : tensor<1024x3072xf32>, tensor<3072x768xf32>
    
    %output = tensor.expand_shape %output_2d [[0, 1], [2]] output_shape [8, 128, 768] : tensor<1024x768xf32> into tensor<8x128x768xf32>
    
    return %output : tensor<8x128x768xf32>
  }

  // Transformer Block
  func.func @transformer_block(%X: tensor<8x128x768xf32>, %W_Q: tensor<768x768xf32>, %W_K: tensor<768x768xf32>, %W_V: tensor<768x768xf32>, %W_O: tensor<768x768xf32>, %W_ff1: tensor<768x3072xf32>, %W_ff2: tensor<3072x768xf32>) -> tensor<8x128x768xf32> {
    // Self-attention sub-layer
    %attn = func.call @self_attention(%X, %W_Q, %W_K, %W_V, %W_O) : (tensor<8x128x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<8x128x768xf32>
    
    // Residual connection: X + attn
    %res1 = nova.add %X, %attn : tensor<8x128x768xf32>, tensor<8x128x768xf32>
    
    // Feed-forward sub-layer
    %ff = func.call @feed_forward(%res1, %W_ff1, %W_ff2) : (tensor<8x128x768xf32>, tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<8x128x768xf32>
    
    // Residual connection: res1 + ff
    %output = nova.add %res1, %ff : tensor<8x128x768xf32>, tensor<8x128x768xf32>
    
    return %output : tensor<8x128x768xf32>
  }

  // Main Function
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : i32
    %f0 = arith.constant 0.0 : f32
    
    // Initialize inputs and weights as tensors
    %inputIdx = tensor.empty() : tensor<8x128xi32>
    %embedding_weight = tensor.empty() : tensor<1000x768xf32>
    %W_Q = tensor.empty() : tensor<768x768xf32>
    %W_K = tensor.empty() : tensor<768x768xf32>
    %W_V = tensor.empty() : tensor<768x768xf32>
    %W_O = tensor.empty() : tensor<768x768xf32>
    %W_ff1 = tensor.empty() : tensor<768x3072xf32>
    %W_ff2 = tensor.empty() : tensor<3072x768xf32>
    
    // Embedding lookup
    %emb = func.call @embedding_lookup(%inputIdx, %embedding_weight) : (tensor<8x128xi32>, tensor<1000x768xf32>) -> tensor<8x128x768xf32>
    
    // Forward through 6 transformer layers
    %layer0 = func.call @transformer_block(%emb, %W_Q, %W_K, %W_V, %W_O, %W_ff1, %W_ff2) : (tensor<8x128x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<8x128x768xf32>
    %layer1 = func.call @transformer_block(%layer0, %W_Q, %W_K, %W_V, %W_O, %W_ff1, %W_ff2) : (tensor<8x128x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<8x128x768xf32>
    %layer2 = func.call @transformer_block(%layer1, %W_Q, %W_K, %W_V, %W_O, %W_ff1, %W_ff2) : (tensor<8x128x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<8x128x768xf32>
    %layer3 = func.call @transformer_block(%layer2, %W_Q, %W_K, %W_V, %W_O, %W_ff1, %W_ff2) : (tensor<8x128x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<8x128x768xf32>
    %layer4 = func.call @transformer_block(%layer3, %W_Q, %W_K, %W_V, %W_O, %W_ff1, %W_ff2) : (tensor<8x128x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<8x128x768xf32>
    %layer5 = func.call @transformer_block(%layer4, %W_Q, %W_K, %W_V, %W_O, %W_ff1, %W_ff2) : (tensor<8x128x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x3072xf32>, tensor<3072x768xf32>) -> tensor<8x128x768xf32>
    
    return %c0 : i32
  }
}