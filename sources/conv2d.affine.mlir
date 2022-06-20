#map = affine_map<(d0, d1) -> (d0 + d1)>
memref.global "private" constant @__constant_64x1x3x3xf32
  : memref<64x1x3x3xf32> = dense<1.000000e+00>
func @forward(%i0: memref<1x1x11x11xf32>) -> memref<1x64x9x9xf32> {
  ...
  affine.for %i1 = 0 to 1 :
    affine.for %i2 = 0 to 64 :
      affine.for %i3 = 0 to 9 :
        affine.for %i4 = 0 to 9 :
          affine.store %cst, %1[%i1, %i2, %i3, %i4]
            : memref<1x64x9x9xf32>

  memref.copy %1, %2 : memref<1x64x9x9xf32> to memref<1x64x9x9xf32>
  affine.for %i1 = 0 to 1 :
    affine.for %i2 = 0 to 64 :
      affine.for %i3 = 0 to 9 :
        affine.for %i4 = 0 to 9 :
          affine.for %i5 = 0 to 1 :
            affine.for %i6 = 0 to 3 :
              affine.for %i7 = 0 to 3 :
                %3 = affine.apply #map(%i3, %i6)
                %4 = affine.apply #map(%i4, %i7)
                %5 = affine.load %i0[%i1, %i5, %3, %4]
                  : memref<1x1x11x11xf32>
                %6 = affine.load %0[%i2, %i5, %i6, %i7]
                  : memref<64x1x3x3xf32>
                %7 = affine.load %2[%i1, %i2, %i3, %i4]
                  : memref<1x64x9x9xf32>
                %8 = arith.mulf %5, %6 : f32
                %9 = arith.addf %7, %8 : f32
                affine.store %9, %2[%i1, %i2, %i3, %i4]
                  : memref<1x64x9x9xf32>

  return %2 : memref<1x64x9x9xf32>
}