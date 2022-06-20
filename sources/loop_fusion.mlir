%weight = memref.get_global @c1 : memref<64x1x3x3xf32>
%bias = memref.get_global @c2 : memref<64xf32>
%tmp = memref.alloca() : memref<1x64x9x9xf32>
scf.for %i1 = %c0 to %c1 step %c1 :
  scf.for %i2 = %c0 to %c64 step %c1 :
    scf.for %i3 = %c0 to %c9 step %c1 :
      scf.for %i4 = %c0 to %c9 step %c1 :
        %3 = memref.load %bias[%i2] 
        memref.store %3, %tmp[%i1, %i2, %i3, %i4] 

scf.for %i1 = %c0 to %c1 step %c1 :
  scf.for %i2 = %c0 to %c64 step %c1 :
    scf.for %i3 = %c0 to %c9 step %c1 :
      scf.for %i4 = %c0 to %c9 step %c1 :
        scf.for %i5 = %c0 to %c1 step %c1 :
          scf.for %i6 = %c0 to %c3 step %c1 :
            scf.for %i7 = %c0 to %c3 step %c1 :
              %3 = arith.addi %i3, %i6 : i32
              %4 = arith.addi %i4, %i7 : i32
              %5 = memref.load %inp[%i1, %i5, %3, %4] 
              %6 = memref.load %weight[%i2, %i5, %i6, %i7] 
              %7 = memref.load %tmp[%i1, %i2, %i3, %i4] 
              %8 = arith.mulf %5, %6 : f16
              %9 = arith.addf %7, %8 : f16
              memref.store %9, %tmp[%i1, %i2, %i3, %i4] 