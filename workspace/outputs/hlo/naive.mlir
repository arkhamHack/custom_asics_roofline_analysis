module @jit_naive_attention attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}, %arg1: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}, %arg2: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}) -> (tensor<2x8x512x64xf16> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x8x512x512xf16> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0:2 = call @naive_attention(%arg0, %arg1, %arg2) : (tensor<2x8x512x64xf16>, tensor<2x8x512x64xf16>, tensor<2x8x512x64xf16>) -> (tensor<2x8x512x64xf16>, tensor<2x8x512x512xf16>)
    return %0#0, %0#1 : tensor<2x8x512x64xf16>, tensor<2x8x512x512xf16>
  }
  func.func private @naive_attention(%arg0: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}, %arg1: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}, %arg2: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}) -> (tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}, tensor<2x8x512x512xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %0 = stablehlo.sqrt %cst : tensor<f32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = stablehlo.divide %cst_0, %0 : tensor<f32>
    %2 = stablehlo.convert %arg0 : (tensor<2x8x512x64xf16>) -> tensor<2x8x512x64xf32>
    %3 = stablehlo.convert %arg1 : (tensor<2x8x512x64xf16>) -> tensor<2x8x512x64xf32>
    %4 = stablehlo.dot_general %2, %3, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x8x512x64xf32>, tensor<2x8x512x64xf32>) -> tensor<2x8x512x512xf16>
    %5 = stablehlo.convert %1 : (tensor<f32>) -> tensor<f16>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f16>) -> tensor<2x8x512x512xf16>
    %7 = stablehlo.multiply %4, %6 : tensor<2x8x512x512xf16>
    %cst_1 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %8 = stablehlo.reduce(%7 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<2x8x512x512xf16>, tensor<f16>) -> tensor<2x8x512xf16>
    %cst_2 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %9 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f16>) -> tensor<2x8x512xf16>
    %10 = stablehlo.maximum %9, %8 : tensor<2x8x512xf16>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<2x8x512xf16>) -> tensor<2x8x512x1xf16>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2, 3] : (tensor<2x8x512x1xf16>) -> tensor<2x8x512x512xf16>
    %13 = stablehlo.subtract %7, %12 : tensor<2x8x512x512xf16>
    %14 = stablehlo.exponential %13 : tensor<2x8x512x512xf16>
    %15 = stablehlo.convert %14 : (tensor<2x8x512x512xf16>) -> tensor<2x8x512x512xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.reduce(%15 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<2x8x512x512xf32>, tensor<f32>) -> tensor<2x8x512xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<2x8x512xf32>) -> tensor<2x8x512x1xf32>
    %18 = stablehlo.convert %17 : (tensor<2x8x512x1xf32>) -> tensor<2x8x512x1xf16>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2, 3] : (tensor<2x8x512x1xf16>) -> tensor<2x8x512x512xf16>
    %20 = stablehlo.divide %14, %19 : tensor<2x8x512x512xf16>
    %21 = stablehlo.convert %20 : (tensor<2x8x512x512xf16>) -> tensor<2x8x512x512xf32>
    %22 = stablehlo.convert %arg2 : (tensor<2x8x512x64xf16>) -> tensor<2x8x512x64xf32>
    %23 = stablehlo.dot_general %21, %22, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x8x512x512xf32>, tensor<2x8x512x64xf32>) -> tensor<2x8x512x64xf16>
    return %23, %20 : tensor<2x8x512x64xf16>, tensor<2x8x512x512xf16>
  }
}
