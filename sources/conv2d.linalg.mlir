func @forward(%arg0: tensor<1x1x11x11xf32>) -> tensor<1x64x9x9xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c1_i64 = arith.constant 1 : i64
  %cst_0 = arith.constant dense<1.000000e+00> : tensor<64x1x3x3xf32>
  %0 = arith.cmpi eq, %c1_i64, %c1_i64 : i64
  %1 = linalg.init_tensor [1, 64, 9, 9] : tensor<1x64x9x9xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x64x9x9xf32>)
    -> tensor<1x64x9x9xf32>
  %3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : vector<2xi64>,
    strides = dense<1> : vector<2xi64>
  } ins(%arg0, %cst_0 : tensor<1x1x11x11xf32>, tensor<64x1x3x3xf32>)
    outs(%2 : tensor<1x64x9x9xf32>) -> tensor<1x64x9x9xf32>
  return %3 : tensor<1x64x9x9xf32>
}