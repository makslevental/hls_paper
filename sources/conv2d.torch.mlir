func @forward(%arg0: !torch.vtensor<[1,1,11,11],f32>)
    -> !torch.vtensor<[1,64,9,9],f32> {
  %none = torch.constant.none
  %0 = torch.vtensor.literal(
      dense<"#DEADBEEF"> : tensor<64x1x3x3xf32>
    ) : !torch.vtensor<[64,1,3,3],f32>
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %1 = torch.prim.ListConstruct %int1, %int1
    : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0
    : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.conv2d %arg0, %0, %none, %1, %2, %1, %int1
    : !torch.vtensor<[1,1,11,11],f32>,
      !torch.vtensor<[64,1,3,3],f32>,
      !torch.none, !torch.list<int>, !torch.list<int>,
      !torch.list<int>, !torch.int
      ->  !torch.vtensor<[1,64,9,9],f32>
  return %3 : !torch.vtensor<[1,64,9,9],f32>
}