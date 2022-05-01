#map = affine_map<(d0, d1) -> (d0 + d1)>
memref.global "private" constant @__constant_64x1x3x3xf32
  : memref<64x1x3x3xf32> = dense<1.000000e+00>
func @forward(%arg0: memref<1x1x11x11xf32>) -> memref<1x64x9x9xf32> {
  ...
  affine.for %arg1 = 0 to 1 :
    affine.for %arg2 = 0 to 64 :
      affine.for %arg3 = 0 to 9 :
        affine.for %arg4 = 0 to 9 :
          affine.store %cst, %1[%arg1, %arg2, %arg3, %arg4]
            : memref<1x64x9x9xf32>

  memref.copy %1, %2 : memref<1x64x9x9xf32> to memref<1x64x9x9xf32>
  affine.for %arg1 = 0 to 1 :
    affine.for %arg2 = 0 to 64 :
      affine.for %arg3 = 0 to 9 :
        affine.for %arg4 = 0 to 9 :
          affine.for %arg5 = 0 to 1 :
            affine.for %arg6 = 0 to 3 :
              affine.for %arg7 = 0 to 3 :
                %3 = affine.apply #map(%arg3, %arg6)
                %4 = affine.apply #map(%arg4, %arg7)
                %5 = affine.load %arg0[%arg1, %arg5, %3, %4]
                  : memref<1x1x11x11xf32>
                %6 = affine.load %0[%arg2, %arg5, %arg6, %arg7]
                  : memref<64x1x3x3xf32>
                %7 = affine.load %2[%arg1, %arg2, %arg3, %arg4]
                  : memref<1x64x9x9xf32>
                %8 = arith.mulf %5, %6 : f32
                %9 = arith.addf %7, %8 : f32
                affine.store %9, %2[%arg1, %arg2, %arg3, %arg4]
                  : memref<1x64x9x9xf32>

  return %2 : memref<1x64x9x9xf32>
}