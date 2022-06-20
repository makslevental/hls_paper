memref.global "private" constant @__constant_64x1x3x3xf32
  : memref<64x1x3x3xf32> = dense<"0xDEADBEEF">
memref.global "private" constant @__constant_64xf32
  : memref<64xf32> = dense<[-0.0887415111, ..., -0.260554731]>
func @forward(%i0: memref<1x1x11x11xf32>) {
  %true = arith.constant true
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c9 = arith.constant 9 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %0 = memref.get_global @__constant_64x1x3x3xf32
    : memref<64x1x3x3xf32>
  %1 = memref.get_global @__constant_64xf32
    : memref<64xf32>
  %2 = memref.alloca() : memref<1x64x9x9xf32>
  scf.for %i1 = %c0 to %c1 step %c1 :
    scf.for %i2 = %c0 to %c64 step %c1 :
      scf.for %i3 = %c0 to %c9 step %c1 :
        scf.for %i4 = %c0 to %c9 step %c1 :
          %3 = memref.load %1[%i2] : memref<64xf32>
          memref.store %3, %2[%i1, %i2, %i3, %i4]
            : memref<1x64x9x9xf32>

  scf.for %i1 = %c0 to %c1 step %c1 :
    scf.for %i2 = %c0 to %c64 step %c1 :
      scf.for %i3 = %c0 to %c9 step %c1 :
        scf.for %i4 = %c0 to %c9 step %c1 :
          scf.for %i5 = %c0 to %c1 step %c1 :
            scf.for %i6 = %c0 to %c3 step %c1 :
              scf.for %i7 = %c0 to %c3 step %c1 :
                %3 = arith.addi %i3, %i6 : index
                %4 = arith.addi %i4, %i7 : index
                %5 = memref.load %i0[%i1, %i5, %3, %4]
                  : memref<1x1x11x11xf32>
                %6 = memref.load %0[%i2, %i5, %i6, %i7]
                  : memref<64x1x3x3xf32>
                %7 = memref.load %2[%i1, %i2, %i3, %i4]
                  : memref<1x64x9x9xf32>
                %8 = arith.mulf %5, %6 : f32
                %9 = arith.addf %7, %8 : f32
                memref.store %9, %2[%i1, %i2, %i3, %i4]
                  : memref<1x64x9x9xf32>

  return
}