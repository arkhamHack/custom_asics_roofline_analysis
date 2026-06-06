module @jit_flash_attention attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}, %arg1: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}, %arg2: tensor<2x8x512x64xf16> {mhlo.layout_mode = "default"}) -> (tensor<2x8x512x64xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<i1>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<2x512xi1>
    %c_0 = stablehlo.constant dense<true> : tensor<i1>
    %1 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i1>) -> tensor<2x512xi1>
    %2 = stablehlo.reshape %arg0 : (tensor<2x8x512x64xf16>) -> tensor<2x8x8x64x64xf16>
    %3 = stablehlo.reshape %arg1 : (tensor<2x8x512x64xf16>) -> tensor<2x8x8x64x64xf16>
    %4 = stablehlo.reshape %arg2 : (tensor<2x8x512x64xf16>) -> tensor<2x8x8x64x64xf16>
    %5 = stablehlo.reshape %1 : (tensor<2x512xi1>) -> tensor<2x8x64xi1>
    %6 = stablehlo.iota dim = 0 : tensor<8xi32>
    %c_1 = stablehlo.constant dense<64> : tensor<i32>
    %7 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %8 = stablehlo.multiply %6, %7 : tensor<8xi32>
    %9 = stablehlo.transpose %2, dims = [0, 2, 1, 3, 4] : (tensor<2x8x8x64x64xf16>) -> tensor<2x8x8x64x64xf16>
    %10 = stablehlo.transpose %9, dims = [1, 0, 2, 3, 4] : (tensor<2x8x8x64x64xf16>) -> tensor<8x2x8x64x64xf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x2x8x64x64xf32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %12:8 = stablehlo.while(%iterArg = %10, %iterArg_3 = %8, %iterArg_4 = %0, %iterArg_5 = %3, %iterArg_6 = %4, %iterArg_7 = %5, %iterArg_8 = %c_2, %iterArg_9 = %11) : tensor<8x2x8x64x64xf16>, tensor<8xi32>, tensor<2x512xi1>, tensor<2x8x8x64x64xf16>, tensor<2x8x8x64x64xf16>, tensor<2x8x64xi1>, tensor<i32>, tensor<8x2x8x64x64xf32>
     cond {
      %c_10 = stablehlo.constant dense<8> : tensor<i32>
      %15 = stablehlo.compare  LT, %iterArg_8, %c_10,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %15 : tensor<i1>
    } do {
      %c_10 = stablehlo.constant dense<0> : tensor<i32>
      %15 = stablehlo.compare  LT, %iterArg_8, %c_10,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %16 = stablehlo.convert %iterArg_8 : tensor<i32>
      %c_11 = stablehlo.constant dense<8> : tensor<i32>
      %17 = stablehlo.add %16, %c_11 : tensor<i32>
      %18 = stablehlo.select %15, %17, %iterArg_8 : tensor<i1>, tensor<i32>
      %c_12 = stablehlo.constant dense<0> : tensor<i32>
      %c_13 = stablehlo.constant dense<0> : tensor<i32>
      %c_14 = stablehlo.constant dense<0> : tensor<i32>
      %c_15 = stablehlo.constant dense<0> : tensor<i32>
      %19 = stablehlo.dynamic_slice %iterArg, %18, %c_12, %c_13, %c_14, %c_15, sizes = [1, 2, 8, 64, 64] : (tensor<8x2x8x64x64xf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x8x64x64xf16>
      %20 = stablehlo.reshape %19 : (tensor<1x2x8x64x64xf16>) -> tensor<2x8x64x64xf16>
      %c_16 = stablehlo.constant dense<0> : tensor<i32>
      %21 = stablehlo.compare  LT, %iterArg_8, %c_16,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %22 = stablehlo.convert %iterArg_8 : tensor<i32>
      %c_17 = stablehlo.constant dense<8> : tensor<i32>
      %23 = stablehlo.add %22, %c_17 : tensor<i32>
      %24 = stablehlo.select %21, %23, %iterArg_8 : tensor<i1>, tensor<i32>
      %25 = stablehlo.dynamic_slice %iterArg_3, %24, sizes = [1] : (tensor<8xi32>, tensor<i32>) -> tensor<1xi32>
      %26 = stablehlo.reshape %25 : (tensor<1xi32>) -> tensor<i32>
      %27 = func.call @None(%iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %20, %26) : (tensor<2x512xi1>, tensor<2x8x8x64x64xf16>, tensor<2x8x8x64x64xf16>, tensor<2x8x64xi1>, tensor<2x8x64x64xf16>, tensor<i32>) -> tensor<2x8x64x64xf32>
      %28 = stablehlo.broadcast_in_dim %27, dims = [1, 2, 3, 4] : (tensor<2x8x64x64xf32>) -> tensor<1x2x8x64x64xf32>
      %c_18 = stablehlo.constant dense<0> : tensor<i32>
      %29 = stablehlo.compare  LT, %iterArg_8, %c_18,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %30 = stablehlo.convert %iterArg_8 : tensor<i32>
      %c_19 = stablehlo.constant dense<8> : tensor<i32>
      %31 = stablehlo.add %30, %c_19 : tensor<i32>
      %32 = stablehlo.select %29, %31, %iterArg_8 : tensor<i1>, tensor<i32>
      %c_20 = stablehlo.constant dense<0> : tensor<i32>
      %c_21 = stablehlo.constant dense<0> : tensor<i32>
      %c_22 = stablehlo.constant dense<0> : tensor<i32>
      %c_23 = stablehlo.constant dense<0> : tensor<i32>
      %33 = stablehlo.dynamic_update_slice %iterArg_9, %28, %32, %c_20, %c_21, %c_22, %c_23 : (tensor<8x2x8x64x64xf32>, tensor<1x2x8x64x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x2x8x64x64xf32>
      %c_24 = stablehlo.constant dense<1> : tensor<i32>
      %34 = stablehlo.add %iterArg_8, %c_24 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_3, %iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %34, %33 : tensor<8x2x8x64x64xf16>, tensor<8xi32>, tensor<2x512xi1>, tensor<2x8x8x64x64xf16>, tensor<2x8x8x64x64xf16>, tensor<2x8x64xi1>, tensor<i32>, tensor<8x2x8x64x64xf32>
    }
    %13 = stablehlo.transpose %12#7, dims = [1, 2, 0, 3, 4] : (tensor<8x2x8x64x64xf32>) -> tensor<2x8x8x64x64xf32>
    %14 = stablehlo.reshape %13 : (tensor<2x8x8x64x64xf32>) -> tensor<2x8x512x64xf32>
    return %14 : tensor<2x8x512x64xf32>
  }
  func.func private @None(%arg0: tensor<2x512xi1>, %arg1: tensor<2x8x8x64x64xf16>, %arg2: tensor<2x8x8x64x64xf16>, %arg3: tensor<2x8x64xi1>, %arg4: tensor<2x8x64x64xf16>, %arg5: tensor<i32>) -> tensor<2x8x64x64xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.compare  LT, %arg5, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_2 = stablehlo.constant dense<512> : tensor<i32>
    %4 = stablehlo.add %arg5, %c_2 : tensor<i32>
    %5 = stablehlo.select %3, %4, %arg5 : tensor<i1>, tensor<i32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], start_index_map = [1]>, indices_are_sorted = true, slice_sizes = array<i64: 2, 64>}> : (tensor<2x512xi1>, tensor<1xi32>) -> tensor<2x64xi1>
    %8 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<64xf32>) -> tensor<8x64xf32>
    %9 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<64xf32>) -> tensor<8x64xf32>
    %10 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<64x64xf32>) -> tensor<8x64x64xf32>
    %11 = stablehlo.transpose %arg1, dims = [0, 2, 1, 3, 4] : (tensor<2x8x8x64x64xf16>) -> tensor<2x8x8x64x64xf16>
    %12 = stablehlo.transpose %arg2, dims = [0, 2, 1, 3, 4] : (tensor<2x8x8x64x64xf16>) -> tensor<2x8x8x64x64xf16>
    %13 = stablehlo.broadcast_in_dim %8, dims = [1, 2] : (tensor<8x64xf32>) -> tensor<2x8x64xf32>
    %14 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<8x64xf32>) -> tensor<2x8x64xf32>
    %15 = stablehlo.broadcast_in_dim %10, dims = [1, 2, 3] : (tensor<8x64x64xf32>) -> tensor<2x8x64x64xf32>
    %16 = stablehlo.transpose %11, dims = [1, 0, 2, 3, 4] : (tensor<2x8x8x64x64xf16>) -> tensor<8x2x8x64x64xf16>
    %17 = stablehlo.transpose %12, dims = [1, 0, 2, 3, 4] : (tensor<2x8x8x64x64xf16>) -> tensor<8x2x8x64x64xf16>
    %18 = stablehlo.transpose %arg3, dims = [1, 0, 2] : (tensor<2x8x64xi1>) -> tensor<8x2x64xi1>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %19:9 = stablehlo.while(%iterArg = %16, %iterArg_6 = %17, %iterArg_7 = %18, %iterArg_8 = %arg4, %iterArg_9 = %7, %iterArg_10 = %c_3, %iterArg_11 = %13, %iterArg_12 = %14, %iterArg_13 = %15) : tensor<8x2x8x64x64xf16>, tensor<8x2x8x64x64xf16>, tensor<8x2x64xi1>, tensor<2x8x64x64xf16>, tensor<2x64xi1>, tensor<i32>, tensor<2x8x64xf32>, tensor<2x8x64xf32>, tensor<2x8x64x64xf32>
     cond {
      %c_14 = stablehlo.constant dense<8> : tensor<i32>
      %28 = stablehlo.compare  LT, %iterArg_10, %c_14,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %28 : tensor<i1>
    } do {
      %c_14 = stablehlo.constant dense<0> : tensor<i32>
      %28 = stablehlo.compare  LT, %iterArg_10, %c_14,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %29 = stablehlo.convert %iterArg_10 : tensor<i32>
      %c_15 = stablehlo.constant dense<8> : tensor<i32>
      %30 = stablehlo.add %29, %c_15 : tensor<i32>
      %31 = stablehlo.select %28, %30, %iterArg_10 : tensor<i1>, tensor<i32>
      %c_16 = stablehlo.constant dense<0> : tensor<i32>
      %c_17 = stablehlo.constant dense<0> : tensor<i32>
      %c_18 = stablehlo.constant dense<0> : tensor<i32>
      %c_19 = stablehlo.constant dense<0> : tensor<i32>
      %32 = stablehlo.dynamic_slice %iterArg, %31, %c_16, %c_17, %c_18, %c_19, sizes = [1, 2, 8, 64, 64] : (tensor<8x2x8x64x64xf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x8x64x64xf16>
      %33 = stablehlo.reshape %32 : (tensor<1x2x8x64x64xf16>) -> tensor<2x8x64x64xf16>
      %c_20 = stablehlo.constant dense<0> : tensor<i32>
      %34 = stablehlo.compare  LT, %iterArg_10, %c_20,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %35 = stablehlo.convert %iterArg_10 : tensor<i32>
      %c_21 = stablehlo.constant dense<8> : tensor<i32>
      %36 = stablehlo.add %35, %c_21 : tensor<i32>
      %37 = stablehlo.select %34, %36, %iterArg_10 : tensor<i1>, tensor<i32>
      %c_22 = stablehlo.constant dense<0> : tensor<i32>
      %c_23 = stablehlo.constant dense<0> : tensor<i32>
      %c_24 = stablehlo.constant dense<0> : tensor<i32>
      %c_25 = stablehlo.constant dense<0> : tensor<i32>
      %38 = stablehlo.dynamic_slice %iterArg_6, %37, %c_22, %c_23, %c_24, %c_25, sizes = [1, 2, 8, 64, 64] : (tensor<8x2x8x64x64xf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x8x64x64xf16>
      %39 = stablehlo.reshape %38 : (tensor<1x2x8x64x64xf16>) -> tensor<2x8x64x64xf16>
      %c_26 = stablehlo.constant dense<0> : tensor<i32>
      %40 = stablehlo.compare  LT, %iterArg_10, %c_26,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %41 = stablehlo.convert %iterArg_10 : tensor<i32>
      %c_27 = stablehlo.constant dense<8> : tensor<i32>
      %42 = stablehlo.add %41, %c_27 : tensor<i32>
      %43 = stablehlo.select %40, %42, %iterArg_10 : tensor<i1>, tensor<i32>
      %c_28 = stablehlo.constant dense<0> : tensor<i32>
      %c_29 = stablehlo.constant dense<0> : tensor<i32>
      %44 = stablehlo.dynamic_slice %iterArg_7, %43, %c_28, %c_29, sizes = [1, 2, 64] : (tensor<8x2x64xi1>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x64xi1>
      %45 = stablehlo.reshape %44 : (tensor<1x2x64xi1>) -> tensor<2x64xi1>
      %46:3 = func.call @None_0(%iterArg_8, %iterArg_9, %iterArg_11, %iterArg_12, %iterArg_13, %33, %39, %45) : (tensor<2x8x64x64xf16>, tensor<2x64xi1>, tensor<2x8x64xf32>, tensor<2x8x64xf32>, tensor<2x8x64x64xf32>, tensor<2x8x64x64xf16>, tensor<2x8x64x64xf16>, tensor<2x64xi1>) -> (tensor<2x8x64xf32>, tensor<2x8x64xf32>, tensor<2x8x64x64xf32>)
      %c_30 = stablehlo.constant dense<1> : tensor<i32>
      %47 = stablehlo.add %iterArg_10, %c_30 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_6, %iterArg_7, %iterArg_8, %iterArg_9, %47, %46#0, %46#1, %46#2 : tensor<8x2x8x64x64xf16>, tensor<8x2x8x64x64xf16>, tensor<8x2x64xi1>, tensor<2x8x64x64xf16>, tensor<2x64xi1>, tensor<i32>, tensor<2x8x64xf32>, tensor<2x8x64xf32>, tensor<2x8x64x64xf32>
    }
    %20 = stablehlo.broadcast_in_dim %19#7, dims = [0, 1, 2] : (tensor<2x8x64xf32>) -> tensor<2x8x64x1xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<2x8x64x1xf32>
    %22 = stablehlo.compare  GT, %20, %21,  FLOAT : (tensor<2x8x64x1xf32>, tensor<2x8x64x1xf32>) -> tensor<2x8x64x1xi1>
    %23 = stablehlo.broadcast_in_dim %19#7, dims = [0, 1, 2] : (tensor<2x8x64xf32>) -> tensor<2x8x64x1xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2, 3] : (tensor<2x8x64x1xf32>) -> tensor<2x8x64x64xf32>
    %25 = stablehlo.divide %19#8, %24 : tensor<2x8x64x64xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %27 = call @_where_4(%22, %25, %26) : (tensor<2x8x64x1xi1>, tensor<2x8x64x64xf32>, tensor<64x64xf32>) -> tensor<2x8x64x64xf32>
    return %27 : tensor<2x8x64x64xf32>
  }
  func.func private @None_0(%arg0: tensor<2x8x64x64xf16>, %arg1: tensor<2x64xi1>, %arg2: tensor<2x8x64xf32>, %arg3: tensor<2x8x64xf32>, %arg4: tensor<2x8x64x64xf32>, %arg5: tensor<2x8x64x64xf16>, %arg6: tensor<2x8x64x64xf16>, %arg7: tensor<2x64xi1>) -> (tensor<2x8x64xf32>, tensor<2x8x64xf32>, tensor<2x8x64x64xf32>) {
    %0 = stablehlo.convert %arg0 : (tensor<2x8x64x64xf16>) -> tensor<2x8x64x64xf32>
    %1 = stablehlo.convert %arg5 : (tensor<2x8x64x64xf16>) -> tensor<2x8x64x64xf32>
    %2 = stablehlo.dot_general %0, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x8x64x64xf32>, tensor<2x8x64x64xf32>) -> tensor<2x8x64x64xf16>
    %3 = stablehlo.convert %2 : (tensor<2x8x64x64xf16>) -> tensor<2x8x64x64xf32>
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<2x8x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg7, dims = [0, 2] : (tensor<2x64xi1>) -> tensor<2x1x64xi1>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = call @_where(%6, %5, %cst_0) : (tensor<2x1x64xi1>, tensor<2x8x64x64xf32>, tensor<f32>) -> tensor<2x8x64x64xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x64xi1>) -> tensor<2x64x1xi1>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %9 = call @_where_1(%8, %7, %cst_1) : (tensor<2x64x1xi1>, tensor<2x8x64x64xf32>, tensor<f32>) -> tensor<2x8x64x64xf32>
    %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %10 = stablehlo.reduce(%9 init: %cst_2) applies stablehlo.maximum across dimensions = [3] : (tensor<2x8x64x64xf32>, tensor<f32>) -> tensor<2x8x64xf32>
    %11 = stablehlo.convert %arg2 : tensor<2x8x64xf32>
    %12 = stablehlo.maximum %11, %10 : tensor<2x8x64xf32>
    %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<2x8x64xf32>
    %14 = stablehlo.compare  EQ, %arg2, %13,  FLOAT : (tensor<2x8x64xf32>, tensor<2x8x64xf32>) -> tensor<2x8x64xi1>
    %15 = stablehlo.convert %arg2 : tensor<2x8x64xf32>
    %16 = stablehlo.subtract %15, %12 : tensor<2x8x64xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = call @_where_2(%14, %cst_4, %16) : (tensor<2x8x64xi1>, tensor<f32>, tensor<2x8x64xf32>) -> tensor<2x8x64xf32>
    %18 = stablehlo.exponential %17 : tensor<2x8x64xf32>
    %19 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2] : (tensor<2x8x64xf32>) -> tensor<2x8x64x1xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2, 3] : (tensor<2x8x64x1xf32>) -> tensor<2x8x64x64xf32>
    %21 = stablehlo.subtract %9, %20 : tensor<2x8x64x64xf32>
    %22 = stablehlo.exponential %21 : tensor<2x8x64x64xf32>
    %23 = stablehlo.compare  NE, %22, %22,  FLOAT : (tensor<2x8x64x64xf32>, tensor<2x8x64x64xf32>) -> tensor<2x8x64x64xi1>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = call @_where_3(%23, %cst_5, %22) : (tensor<2x8x64x64xi1>, tensor<f32>, tensor<2x8x64x64xf32>) -> tensor<2x8x64x64xf32>
    %25 = stablehlo.multiply %18, %arg3 : tensor<2x8x64xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %26 = stablehlo.reduce(%24 init: %cst_6) applies stablehlo.add across dimensions = [3] : (tensor<2x8x64x64xf32>, tensor<f32>) -> tensor<2x8x64xf32>
    %27 = stablehlo.add %25, %26 : tensor<2x8x64xf32>
    %28 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2] : (tensor<2x8x64xf32>) -> tensor<2x8x64x1xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2, 3] : (tensor<2x8x64x1xf32>) -> tensor<2x8x64x64xf32>
    %30 = stablehlo.multiply %29, %arg4 : tensor<2x8x64x64xf32>
    %31 = stablehlo.convert %24 : tensor<2x8x64x64xf32>
    %32 = stablehlo.convert %arg6 : (tensor<2x8x64x64xf16>) -> tensor<2x8x64x64xf32>
    %33 = stablehlo.dot_general %31, %32, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x8x64x64xf32>, tensor<2x8x64x64xf32>) -> tensor<2x8x64x64xf32>
    %34 = stablehlo.add %30, %33 : tensor<2x8x64x64xf32>
    return %12, %27, %34 : tensor<2x8x64xf32>, tensor<2x8x64xf32>, tensor<2x8x64x64xf32>
  }
  func.func private @_where(%arg0: tensor<2x1x64xi1> {mhlo.layout_mode = "default"}, %arg1: tensor<2x8x64x64xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<f32> {mhlo.layout_mode = "default"}) -> (tensor<2x8x64x64xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.reshape %arg0 : (tensor<2x1x64xi1>) -> tensor<2x64xi1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<2x64xi1>) -> tensor<2x64x64xi1>
    %3 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [0, 2, 3] : (tensor<2x64x64xi1>) -> tensor<2x8x64x64xi1>
    %5 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<64x64xf32>) -> tensor<8x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1, 2, 3] : (tensor<8x64x64xf32>) -> tensor<2x8x64x64xf32>
    %7 = stablehlo.select %4, %arg1, %6 : tensor<2x8x64x64xi1>, tensor<2x8x64x64xf32>
    return %7 : tensor<2x8x64x64xf32>
  }
  func.func private @_where_1(%arg0: tensor<2x64x1xi1> {mhlo.layout_mode = "default"}, %arg1: tensor<2x8x64x64xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<f32> {mhlo.layout_mode = "default"}) -> (tensor<2x8x64x64xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.reshape %arg0 : (tensor<2x64x1xi1>) -> tensor<2x64xi1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<2x64xi1>) -> tensor<2x64x64xi1>
    %3 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [0, 2, 3] : (tensor<2x64x64xi1>) -> tensor<2x8x64x64xi1>
    %5 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<64x64xf32>) -> tensor<8x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1, 2, 3] : (tensor<8x64x64xf32>) -> tensor<2x8x64x64xf32>
    %7 = stablehlo.select %4, %arg1, %6 : tensor<2x8x64x64xi1>, tensor<2x8x64x64xf32>
    return %7 : tensor<2x8x64x64xf32>
  }
  func.func private @_where_2(%arg0: tensor<2x8x64xi1> {mhlo.layout_mode = "default"}, %arg1: tensor<f32> {mhlo.layout_mode = "default"}, %arg2: tensor<2x8x64xf32> {mhlo.layout_mode = "default"}) -> (tensor<2x8x64xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg1 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<64xf32>) -> tensor<8x64xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<8x64xf32>) -> tensor<2x8x64xf32>
    %4 = stablehlo.select %arg0, %3, %arg2 : tensor<2x8x64xi1>, tensor<2x8x64xf32>
    return %4 : tensor<2x8x64xf32>
  }
  func.func private @_where_3(%arg0: tensor<2x8x64x64xi1> {mhlo.layout_mode = "default"}, %arg1: tensor<f32> {mhlo.layout_mode = "default"}, %arg2: tensor<2x8x64x64xf32> {mhlo.layout_mode = "default"}) -> (tensor<2x8x64x64xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg1 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<64x64xf32>) -> tensor<8x64x64xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2, 3] : (tensor<8x64x64xf32>) -> tensor<2x8x64x64xf32>
    %4 = stablehlo.select %arg0, %3, %arg2 : tensor<2x8x64x64xi1>, tensor<2x8x64x64xf32>
    return %4 : tensor<2x8x64x64xf32>
  }
  func.func private @_where_4(%arg0: tensor<2x8x64x1xi1> {mhlo.layout_mode = "default"}, %arg1: tensor<2x8x64x64xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<64x64xf32> {mhlo.layout_mode = "default"}) -> (tensor<2x8x64x64xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<2x8x64x1xi1>) -> tensor<2x8x64xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<2x8x64xi1>) -> tensor<2x8x64x64xi1>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1, 2] : (tensor<64x64xf32>) -> tensor<8x64x64xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2, 3] : (tensor<8x64x64xf32>) -> tensor<2x8x64x64xf32>
    %4 = stablehlo.select %1, %arg1, %3 : tensor<2x8x64x64xi1>, tensor<2x8x64x64xf32>
    return %4 : tensor<2x8x64x64xf32>
  }
}
