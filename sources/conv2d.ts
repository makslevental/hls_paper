graph(%x.1 : Tensor):
  %15 : int[] = prim::Constant[value=[0, 0]]()
  %14 : int[] = prim::Constant[value=[1, 1]]()
  %4 : int = prim::Constant[value=1]()
  %conv1.bias : NoneType = prim::Constant()
  %conv1.weight : Float(64, 1, 3, 3, strides=[9, 9, 3, 1])
    = prim::Constant[value=<Tensor>]()
  %out.1 : Tensor = aten::conv2d(
      %x.1, %conv1.weight, %conv1.bias, %14, %15, %14, %4
    )
  return (%out.1)