module @jit__lambda_ attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}, %arg1: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}, %arg2: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}) -> (tensor<2x8x512x64xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.abs %arg0 : tensor<2x8x512x64xf16>
    %cst = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.maximum across dimensions = [0, 1, 2, 3] : (tensor<2x8x512x64xf16>, tensor<f16>) -> tensor<f16>
    %cst_0 = stablehlo.constant dense<1.270000e+02> : tensor<f16>
    %2 = stablehlo.divide %1, %cst_0 : tensor<f16>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %3 = stablehlo.maximum %2, %cst_1 : tensor<f16>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f16>) -> tensor<2x8x512x64xf16>
    %5 = stablehlo.divide %arg0, %4 : tensor<2x8x512x64xf16>
    %6 = call @round(%5) : (tensor<2x8x512x64xf16>) -> tensor<2x8x512x64xf16>
    %c = stablehlo.constant dense<-127> : tensor<i32>
    %c_2 = stablehlo.constant dense<127> : tensor<i32>
    %7 = call @clip(%6, %c, %c_2) : (tensor<2x8x512x64xf16>, tensor<i32>, tensor<i32>) -> tensor<2x8x512x64xf16>
    %8 = stablehlo.convert %7 : (tensor<2x8x512x64xf16>) -> tensor<2x8x512x64xi8>
    %9 = stablehlo.abs %arg1 : tensor<2x8x512x64xf16>
    %cst_3 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %10 = stablehlo.reduce(%9 init: %cst_3) applies stablehlo.maximum across dimensions = [0, 1, 2, 3] : (tensor<2x8x512x64xf16>, tensor<f16>) -> tensor<f16>
    %cst_4 = stablehlo.constant dense<1.270000e+02> : tensor<f16>
    %11 = stablehlo.divide %10, %cst_4 : tensor<f16>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %12 = stablehlo.maximum %11, %cst_5 : tensor<f16>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f16>) -> tensor<2x8x512x64xf16>
    %14 = stablehlo.divide %arg1, %13 : tensor<2x8x512x64xf16>
    %15 = call @round(%14) : (tensor<2x8x512x64xf16>) -> tensor<2x8x512x64xf16>
    %c_6 = stablehlo.constant dense<-127> : tensor<i32>
    %c_7 = stablehlo.constant dense<127> : tensor<i32>
    %16 = call @clip(%15, %c_6, %c_7) : (tensor<2x8x512x64xf16>, tensor<i32>, tensor<i32>) -> tensor<2x8x512x64xf16>
    %17 = stablehlo.convert %16 : (tensor<2x8x512x64xf16>) -> tensor<2x8x512x64xi8>
    %18 = stablehlo.abs %arg2 : tensor<2x8x512x64xf16>
    %cst_8 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %19 = stablehlo.reduce(%18 init: %cst_8) applies stablehlo.maximum across dimensions = [0, 1, 2, 3] : (tensor<2x8x512x64xf16>, tensor<f16>) -> tensor<f16>
    %cst_9 = stablehlo.constant dense<1.270000e+02> : tensor<f16>
    %20 = stablehlo.divide %19, %cst_9 : tensor<f16>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %21 = stablehlo.maximum %20, %cst_10 : tensor<f16>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<f16>) -> tensor<2x8x512x64xf16>
    %23 = stablehlo.divide %arg2, %22 : tensor<2x8x512x64xf16>
    %24 = call @round(%23) : (tensor<2x8x512x64xf16>) -> tensor<2x8x512x64xf16>
    %c_11 = stablehlo.constant dense<-127> : tensor<i32>
    %c_12 = stablehlo.constant dense<127> : tensor<i32>
    %25 = call @clip(%24, %c_11, %c_12) : (tensor<2x8x512x64xf16>, tensor<i32>, tensor<i32>) -> tensor<2x8x512x64xf16>
    %26 = stablehlo.convert %25 : (tensor<2x8x512x64xf16>) -> tensor<2x8x512x64xi8>
    %27 = stablehlo.reshape %8 : (tensor<2x8x512x64xi8>) -> tensor<2x8x8x64x64xi8>
    %28 = stablehlo.reshape %17 : (tensor<2x8x512x64xi8>) -> tensor<2x8x8x64x64xi8>
    %29 = stablehlo.reshape %26 : (tensor<2x8x512x64xi8>) -> tensor<2x8x8x64x64xi8>
    %30 = stablehlo.transpose %27, dims = [0, 2, 1, 3, 4] : (tensor<2x8x8x64x64xi8>) -> tensor<2x8x8x64x64xi8>
    %31 = stablehlo.transpose %30, dims = [1, 0, 2, 3, 4] : (tensor<2x8x8x64x64xi8>) -> tensor<8x2x8x64x64xi8>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x2x8x64x64xf32>
    %c_14 = stablehlo.constant dense<0> : tensor<i32>
    %33:8 = stablehlo.while(%iterArg = %31, %iterArg_15 = %3, %iterArg_16 = %12, %iterArg_17 = %21, %iterArg_18 = %28, %iterArg_19 = %29, %iterArg_20 = %c_14, %iterArg_21 = %32) : tensor<8x2x8x64x64xi8>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<2x8x8x64x64xi8>, tensor<2x8x8x64x64xi8>, tensor<i32>, tensor<8x2x8x64x64xf32>
     cond {
      %c_22 = stablehlo.constant dense<8> : tensor<i32>
      %37 = stablehlo.compare  LT, %iterArg_20, %c_22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %37 : tensor<i1>
    } do {
      %c_22 = stablehlo.constant dense<0> : tensor<i32>
      %37 = stablehlo.compare  LT, %iterArg_20, %c_22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %38 = stablehlo.convert %iterArg_20 : tensor<i32>
      %c_23 = stablehlo.constant dense<8> : tensor<i32>
      %39 = stablehlo.add %38, %c_23 : tensor<i32>
      %40 = stablehlo.select %37, %39, %iterArg_20 : tensor<i1>, tensor<i32>
      %c_24 = stablehlo.constant dense<0> : tensor<i32>
      %c_25 = stablehlo.constant dense<0> : tensor<i32>
      %c_26 = stablehlo.constant dense<0> : tensor<i32>
      %c_27 = stablehlo.constant dense<0> : tensor<i32>
      %41 = stablehlo.dynamic_slice %iterArg, %40, %c_24, %c_25, %c_26, %c_27, sizes = [1, 2, 8, 64, 64] : (tensor<8x2x8x64x64xi8>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x8x64x64xi8>
      %42 = stablehlo.reshape %41 : (tensor<1x2x8x64x64xi8>) -> tensor<2x8x64x64xi8>
      %43 = func.call @None(%iterArg_15, %iterArg_16, %iterArg_17, %iterArg_18, %iterArg_19, %42) : (tensor<f16>, tensor<f16>, tensor<f16>, tensor<2x8x8x64x64xi8>, tensor<2x8x8x64x64xi8>, tensor<2x8x64x64xi8>) -> tensor<2x8x64x64xf32>
      %44 = stablehlo.broadcast_in_dim %43, dims = [1, 2, 3, 4] : (tensor<2x8x64x64xf32>) -> tensor<1x2x8x64x64xf32>
      %c_28 = stablehlo.constant dense<0> : tensor<i32>
      %45 = stablehlo.compare  LT, %iterArg_20, %c_28,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %46 = stablehlo.convert %iterArg_20 : tensor<i32>
      %c_29 = stablehlo.constant dense<8> : tensor<i32>
      %47 = stablehlo.add %46, %c_29 : tensor<i32>
      %48 = stablehlo.select %45, %47, %iterArg_20 : tensor<i1>, tensor<i32>
      %c_30 = stablehlo.constant dense<0> : tensor<i32>
      %c_31 = stablehlo.constant dense<0> : tensor<i32>
      %c_32 = stablehlo.constant dense<0> : tensor<i32>
      %c_33 = stablehlo.constant dense<0> : tensor<i32>
      %49 = stablehlo.dynamic_update_slice %iterArg_21, %44, %48, %c_30, %c_31, %c_32, %c_33 : (tensor<8x2x8x64x64xf32>, tensor<1x2x8x64x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x2x8x64x64xf32>
      %c_34 = stablehlo.constant dense<1> : tensor<i32>
      %50 = stablehlo.add %iterArg_20, %c_34 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_15, %iterArg_16, %iterArg_17, %iterArg_18, %iterArg_19, %50, %49 : tensor<8x2x8x64x64xi8>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<2x8x8x64x64xi8>, tensor<2x8x8x64x64xi8>, tensor<i32>, tensor<8x2x8x64x64xf32>
    }
    %34 = stablehlo.transpose %33#7, dims = [1, 2, 0, 3, 4] : (tensor<8x2x8x64x64xf32>) -> tensor<2x8x8x64x64xf32>
    %35 = stablehlo.reshape %34 : (tensor<2x8x8x64x64xf32>) -> tensor<2x8x512x64xf32>
    %36 = stablehlo.convert %35 : (tensor<2x8x512x64xf32>) -> tensor<2x8x512x64xf16>
    return %36 : tensor<2x8x512x64xf16>
  }
  func.func private @round(%arg0: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}) -> (tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.round_nearest_even %arg0 : tensor<2x8x512x64xf16>
    return %0 : tensor<2x8x512x64xf16>
  }
  func.func private @clip(%arg0: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}, %arg1: tensor<i32> {mhlo.layout_mode = "default"}, %arg2: tensor<i32> {mhlo.layout_mode = "default"}) -> (tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<f16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f16>) -> tensor<2x8x512x64xf16>
    %2 = stablehlo.maximum %1, %arg0 : tensor<2x8x512x64xf16>
    %3 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f16>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f16>) -> tensor<2x8x512x64xf16>
    %5 = stablehlo.minimum %4, %2 : tensor<2x8x512x64xf16>
    return %5 : tensor<2x8x512x64xf16>
  }
  func.func private @None(%arg0: tensor<f16>, %arg1: tensor<f16>, %arg2: tensor<f16>, %arg3: tensor<2x8x8x64x64xi8>, %arg4: tensor<2x8x8x64x64xi8>, %arg5: tensor<2x8x64x64xi8>) -> tensor<2x8x64x64xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %3 = stablehlo.convert %arg5 : (tensor<2x8x64x64xi8>) -> tensor<2x8x64x64xf32>
    %4 = stablehlo.convert %arg0 : (tensor<f16>) -> tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<2x8x64x64xf32>
    %6 = stablehlo.multiply %3, %5 : tensor<2x8x64x64xf32>
    %7 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<64xf32>) -> tensor<8x64xf32>
    %8 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<64xf32>) -> tensor<8x64xf32>
    %9 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<64x64xf32>) -> tensor<8x64x64xf32>
    %10 = stablehlo.transpose %arg3, dims = [0, 2, 1, 3, 4] : (tensor<2x8x8x64x64xi8>) -> tensor<2x8x8x64x64xi8>
    %11 = stablehlo.transpose %arg4, dims = [0, 2, 1, 3, 4] : (tensor<2x8x8x64x64xi8>) -> tensor<2x8x8x64x64xi8>
    %12 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<8x64xf32>) -> tensor<2x8x64xf32>
    %13 = stablehlo.broadcast_in_dim %8, dims = [1, 2] : (tensor<8x64xf32>) -> tensor<2x8x64xf32>
    %14 = stablehlo.broadcast_in_dim %9, dims = [1, 2, 3] : (tensor<8x64x64xf32>) -> tensor<2x8x64x64xf32>
    %15 = stablehlo.transpose %10, dims = [1, 0, 2, 3, 4] : (tensor<2x8x8x64x64xi8>) -> tensor<8x2x8x64x64xi8>
    %16 = stablehlo.transpose %11, dims = [1, 0, 2, 3, 4] : (tensor<2x8x8x64x64xi8>) -> tensor<8x2x8x64x64xi8>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %17:9 = stablehlo.while(%iterArg = %15, %iterArg_4 = %16, %iterArg_5 = %arg1, %iterArg_6 = %arg2, %iterArg_7 = %6, %iterArg_8 = %c, %iterArg_9 = %12, %iterArg_10 = %13, %iterArg_11 = %14) : tensor<8x2x8x64x64xi8>, tensor<8x2x8x64x64xi8>, tensor<f16>, tensor<f16>, tensor<2x8x64x64xf32>, tensor<i32>, tensor<2x8x64xf32>, tensor<2x8x64xf32>, tensor<2x8x64x64xf32>
     cond {
      %c_12 = stablehlo.constant dense<8> : tensor<i32>
      %26 = stablehlo.compare  LT, %iterArg_8, %c_12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %26 : tensor<i1>
    } do {
      %c_12 = stablehlo.constant dense<0> : tensor<i32>
      %26 = stablehlo.compare  LT, %iterArg_8, %c_12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %27 = stablehlo.convert %iterArg_8 : tensor<i32>
      %c_13 = stablehlo.constant dense<8> : tensor<i32>
      %28 = stablehlo.add %27, %c_13 : tensor<i32>
      %29 = stablehlo.select %26, %28, %iterArg_8 : tensor<i1>, tensor<i32>
      %c_14 = stablehlo.constant dense<0> : tensor<i32>
      %c_15 = stablehlo.constant dense<0> : tensor<i32>
      %c_16 = stablehlo.constant dense<0> : tensor<i32>
      %c_17 = stablehlo.constant dense<0> : tensor<i32>
      %30 = stablehlo.dynamic_slice %iterArg, %29, %c_14, %c_15, %c_16, %c_17, sizes = [1, 2, 8, 64, 64] : (tensor<8x2x8x64x64xi8>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x8x64x64xi8>
      %31 = stablehlo.reshape %30 : (tensor<1x2x8x64x64xi8>) -> tensor<2x8x64x64xi8>
      %c_18 = stablehlo.constant dense<0> : tensor<i32>
      %32 = stablehlo.compare  LT, %iterArg_8, %c_18,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %33 = stablehlo.convert %iterArg_8 : tensor<i32>
      %c_19 = stablehlo.constant dense<8> : tensor<i32>
      %34 = stablehlo.add %33, %c_19 : tensor<i32>
      %35 = stablehlo.select %32, %34, %iterArg_8 : tensor<i1>, tensor<i32>
      %c_20 = stablehlo.constant dense<0> : tensor<i32>
      %c_21 = stablehlo.constant dense<0> : tensor<i32>
      %c_22 = stablehlo.constant dense<0> : tensor<i32>
      %c_23 = stablehlo.constant dense<0> : tensor<i32>
      %36 = stablehlo.dynamic_slice %iterArg_4, %35, %c_20, %c_21, %c_22, %c_23, sizes = [1, 2, 8, 64, 64] : (tensor<8x2x8x64x64xi8>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x8x64x64xi8>
      %37 = stablehlo.reshape %36 : (tensor<1x2x8x64x64xi8>) -> tensor<2x8x64x64xi8>
      %38:3 = func.call @None_0(%iterArg_5, %iterArg_6, %iterArg_7, %iterArg_9, %iterArg_10, %iterArg_11, %31, %37) : (tensor<f16>, tensor<f16>, tensor<2x8x64x64xf32>, tensor<2x8x64xf32>, tensor<2x8x64xf32>, tensor<2x8x64x64xf32>, tensor<2x8x64x64xi8>, tensor<2x8x64x64xi8>) -> (tensor<2x8x64xf32>, tensor<2x8x64xf32>, tensor<2x8x64x64xf32>)
      %c_24 = stablehlo.constant dense<1> : tensor<i32>
      %39 = stablehlo.add %iterArg_8, %c_24 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %39, %38#0, %38#1, %38#2 : tensor<8x2x8x64x64xi8>, tensor<8x2x8x64x64xi8>, tensor<f16>, tensor<f16>, tensor<2x8x64x64xf32>, tensor<i32>, tensor<2x8x64xf32>, tensor<2x8x64xf32>, tensor<2x8x64x64xf32>
    }
    %18 = stablehlo.broadcast_in_dim %17#7, dims = [0, 1, 2] : (tensor<2x8x64xf32>) -> tensor<2x8x64x1xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<2x8x64x1xf32>
    %20 = stablehlo.compare  GT, %18, %19,  FLOAT : (tensor<2x8x64x1xf32>, tensor<2x8x64x1xf32>) -> tensor<2x8x64x1xi1>
    %21 = stablehlo.broadcast_in_dim %17#7, dims = [0, 1, 2] : (tensor<2x8x64xf32>) -> tensor<2x8x64x1xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2, 3] : (tensor<2x8x64x1xf32>) -> tensor<2x8x64x64xf32>
    %23 = stablehlo.divide %17#8, %22 : tensor<2x8x64x64xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %25 = call @_where_2(%20, %23, %24) : (tensor<2x8x64x1xi1>, tensor<2x8x64x64xf32>, tensor<64x64xf32>) -> tensor<2x8x64x64xf32>
    return %25 : tensor<2x8x64x64xf32>
  }
  func.func private @None_0(%arg0: tensor<f16>, %arg1: tensor<f16>, %arg2: tensor<2x8x64x64xf32>, %arg3: tensor<2x8x64xf32>, %arg4: tensor<2x8x64xf32>, %arg5: tensor<2x8x64x64xf32>, %arg6: tensor<2x8x64x64xi8>, %arg7: tensor<2x8x64x64xi8>) -> (tensor<2x8x64xf32>, tensor<2x8x64xf32>, tensor<2x8x64x64xf32>) {
    %0 = stablehlo.convert %arg6 : (tensor<2x8x64x64xi8>) -> tensor<2x8x64x64xf32>
    %1 = stablehlo.convert %arg0 : (tensor<f16>) -> tensor<f32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<2x8x64x64xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<2x8x64x64xf32>
    %4 = stablehlo.convert %arg7 : (tensor<2x8x64x64xi8>) -> tensor<2x8x64x64xf32>
    %5 = stablehlo.convert %arg1 : (tensor<f16>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<2x8x64x64xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<2x8x64x64xf32>
    %8 = stablehlo.dot_general %arg2, %3, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x8x64x64xf32>, tensor<2x8x64x64xf32>) -> tensor<2x8x64x64xf32>
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8x64x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<2x8x64x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<2x8x64x64xf32>, tensor<f32>) -> tensor<2x8x64xf32>
    %12 = stablehlo.convert %arg3 : tensor<2x8x64xf32>
    %13 = stablehlo.maximum %12, %11 : tensor<2x8x64xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2x8x64xf32>
    %15 = stablehlo.compare  EQ, %arg3, %14,  FLOAT : (tensor<2x8x64xf32>, tensor<2x8x64xf32>) -> tensor<2x8x64xi1>
    %16 = stablehlo.convert %arg3 : tensor<2x8x64xf32>
    %17 = stablehlo.subtract %16, %13 : tensor<2x8x64xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = call @_where(%15, %cst_2, %17) : (tensor<2x8x64xi1>, tensor<f32>, tensor<2x8x64xf32>) -> tensor<2x8x64xf32>
    %19 = stablehlo.exponential %18 : tensor<2x8x64xf32>
    %20 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<2x8x64xf32>) -> tensor<2x8x64x1xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2, 3] : (tensor<2x8x64x1xf32>) -> tensor<2x8x64x64xf32>
    %22 = stablehlo.subtract %10, %21 : tensor<2x8x64x64xf32>
    %23 = stablehlo.exponential %22 : tensor<2x8x64x64xf32>
    %24 = stablehlo.compare  NE, %23, %23,  FLOAT : (tensor<2x8x64x64xf32>, tensor<2x8x64x64xf32>) -> tensor<2x8x64x64xi1>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = call @_where_1(%24, %cst_3, %23) : (tensor<2x8x64x64xi1>, tensor<f32>, tensor<2x8x64x64xf32>) -> tensor<2x8x64x64xf32>
    %26 = stablehlo.multiply %19, %arg4 : tensor<2x8x64xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.reduce(%25 init: %cst_4) applies stablehlo.add across dimensions = [3] : (tensor<2x8x64x64xf32>, tensor<f32>) -> tensor<2x8x64xf32>
    %28 = stablehlo.add %26, %27 : tensor<2x8x64xf32>
    %29 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2] : (tensor<2x8x64xf32>) -> tensor<2x8x64x1xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2, 3] : (tensor<2x8x64x1xf32>) -> tensor<2x8x64x64xf32>
    %31 = stablehlo.multiply %30, %arg5 : tensor<2x8x64x64xf32>
    %32 = stablehlo.dot_general %25, %7, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x8x64x64xf32>, tensor<2x8x64x64xf32>) -> tensor<2x8x64x64xf32>
    %33 = stablehlo.add %31, %32 : tensor<2x8x64x64xf32>
    return %13, %28, %33 : tensor<2x8x64xf32>, tensor<2x8x64xf32>, tensor<2x8x64x64xf32>
  }
  func.func private @_where(%arg0: tensor<2x8x64xi1> {mhlo.layout_mode = "default"}, %arg1: tensor<f32> {mhlo.layout_mode = "default"}, %arg2: tensor<2x8x64xf32> {mhlo.layout_mode = "default"}) -> (tensor<2x8x64xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg1 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<64xf32>) -> tensor<8x64xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<8x64xf32>) -> tensor<2x8x64xf32>
    %4 = stablehlo.select %arg0, %3, %arg2 : tensor<2x8x64xi1>, tensor<2x8x64xf32>
    return %4 : tensor<2x8x64xf32>
  }
  func.func private @_where_1(%arg0: tensor<2x8x64x64xi1> {mhlo.layout_mode = "default"}, %arg1: tensor<f32> {mhlo.layout_mode = "default"}, %arg2: tensor<2x8x64x64xf32> {mhlo.layout_mode = "default"}) -> (tensor<2x8x64x64xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg1 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<64x64xf32>) -> tensor<8x64x64xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2, 3] : (tensor<8x64x64xf32>) -> tensor<2x8x64x64xf32>
    %4 = stablehlo.select %arg0, %3, %arg2 : tensor<2x8x64x64xi1>, tensor<2x8x64x64xf32>
    return %4 : tensor<2x8x64x64xf32>
  }
  func.func private @_where_2(%arg0: tensor<2x8x64x1xi1> {mhlo.layout_mode = "default"}, %arg1: tensor<2x8x64x64xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<64x64xf32> {mhlo.layout_mode = "default"}) -> (tensor<2x8x64x64xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<2x8x64x1xi1>) -> tensor<2x8x64xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<2x8x64xi1>) -> tensor<2x8x64x64xi1>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1, 2] : (tensor<64x64xf32>) -> tensor<8x64x64xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2, 3] : (tensor<8x64x64xf32>) -> tensor<2x8x64x64xf32>
    %4 = stablehlo.select %1, %arg1, %3 : tensor<2x8x64x64xi1>, tensor<2x8x64x64xf32>
    return %4 : tensor<2x8x64x64xf32>
  }
}
