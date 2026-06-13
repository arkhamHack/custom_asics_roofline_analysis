module @jit__lambda_ attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}, %arg1: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}, %arg2: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}) -> (tensor<2x8x512x64xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<512> : tensor<2xi32>
    %cst = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %0 = stablehlo.sqrt %cst : tensor<f32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = stablehlo.divide %cst_0, %0 : tensor<f32>
    %2 = stablehlo.iota dim = 0 : tensor<512xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<512xi32>) -> tensor<1x512xi32>
    %4 = stablehlo.broadcast_in_dim %c, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x512xi32>) -> tensor<2x512xi32>
    %6 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<2x1xi32>) -> tensor<2x512xi32>
    %7 = stablehlo.compare  LT, %5, %6,  SIGNED : (tensor<2x512xi32>, tensor<2x512xi32>) -> tensor<2x512xi1>
    %8 = stablehlo.convert %arg0 : (tensor<2x8x512x64xf16>) -> tensor<2x8x512x64xf32>
    %9 = stablehlo.convert %arg1 : (tensor<2x8x512x64xf16>) -> tensor<2x8x512x64xf32>
    %10 = stablehlo.dot_general %8, %9, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x8x512x64xf32>, tensor<2x8x512x64xf32>) -> tensor<2x8x512x512xf16>
    %11 = stablehlo.convert %10 : (tensor<2x8x512x512xf16>) -> tensor<2x8x512x512xf32>
    %12 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<2x8x512x512xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<2x8x512x512xf32>
    %14 = stablehlo.broadcast_in_dim %7, dims = [0, 3] : (tensor<2x512xi1>) -> tensor<2x1x1x512xi1>
    %cst_1 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %15 = call @_where(%14, %13, %cst_1) : (tensor<2x1x1x512xi1>, tensor<2x8x512x512xf32>, tensor<f32>) -> tensor<2x8x512x512xf32>
    %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %16 = stablehlo.reduce(%15 init: %cst_2) applies stablehlo.maximum across dimensions = [3] : (tensor<2x8x512x512xf32>, tensor<f32>) -> tensor<2x8x512xf32>
    %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<2x8x512xf32>
    %18 = stablehlo.maximum %17, %16 : tensor<2x8x512xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2] : (tensor<2x8x512xf32>) -> tensor<2x8x512x1xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2, 3] : (tensor<2x8x512x1xf32>) -> tensor<2x8x512x512xf32>
    %21 = stablehlo.subtract %15, %20 : tensor<2x8x512x512xf32>
    %22 = stablehlo.exponential %21 : tensor<2x8x512x512xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %23 = stablehlo.reduce(%22 init: %cst_4) applies stablehlo.add across dimensions = [3] : (tensor<2x8x512x512xf32>, tensor<f32>) -> tensor<2x8x512xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2] : (tensor<2x8x512xf32>) -> tensor<2x8x512x1xf32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [0, 1, 2, 3] : (tensor<2x8x512x1xf32>) -> tensor<2x8x512x512xf32>
    %26 = stablehlo.divide %22, %25 : tensor<2x8x512x512xf32>
    %27 = stablehlo.convert %26 : tensor<2x8x512x512xf32>
    %28 = stablehlo.convert %arg2 : (tensor<2x8x512x64xf16>) -> tensor<2x8x512x64xf32>
    %29 = stablehlo.dot_general %27, %28, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x8x512x512xf32>, tensor<2x8x512x64xf32>) -> tensor<2x8x512x64xf32>
    %30 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<2x512xi1>) -> tensor<2x1x512x1xi1>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %31 = call @_where_0(%30, %29, %cst_5) : (tensor<2x1x512x1xi1>, tensor<2x8x512x64xf32>, tensor<f32>) -> tensor<2x8x512x64xf32>
    return %31 : tensor<2x8x512x64xf32>
  }
  func.func private @_where(%arg0: tensor<2x1x1x512xi1> {mhlo.layout_mode = "default"}, %arg1: tensor<2x8x512x512xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<f32> {mhlo.layout_mode = "default"}) -> (tensor<2x8x512x512xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<2x1x1x512xi1>) -> tensor<2x512xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 3] : (tensor<2x512xi1>) -> tensor<2x8x512x512xi1>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x8x512x512xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<2x8x512x512xi1>, tensor<2x8x512x512xf32>
    return %3 : tensor<2x8x512x512xf32>
  }
  func.func private @_where_0(%arg0: tensor<2x1x512x1xi1> {mhlo.layout_mode = "default"}, %arg1: tensor<2x8x512x64xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<f32> {mhlo.layout_mode = "default"}) -> (tensor<2x8x512x64xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.reshape %arg0 : (tensor<2x1x512x1xi1>) -> tensor<2x512xi1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<2x512xi1>) -> tensor<2x8x512x64xi1>
    %3 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x8x512x64xf32>
    %4 = stablehlo.select %2, %arg1, %3 : tensor<2x8x512x64xi1>, tensor<2x8x512x64xf32>
    return %4 : tensor<2x8x512x64xf32>
  }
}
