scf.parallel (%i1, %i2, %i3, %i4) = (%c0, %c0, %c0, %c0) 
  to (%c1, %c64, %c9, %c9) step (%c1, %c1, %c1, %c1) :
  scf.for %i5 = %c0 to %c1 step %c1 :
    scf.for %i6 = %c0 to %c3 step %c1 :
      scf.for %i7 = %c0 to %c3 step %c1 :
        %3 = arith.addi %i3, %i6 : i32
        %4 = arith.addi %i4, %i7 : i32
        %5 = memref.load %inp[%i1, %i5, %3, %4] 
        %6 = memref.load %weight[%i2, %i5, %i6, %i7] 
        %7 = memref.load %tmp[%i1, %i2, %i3, %i4] 
        %8 = arith.mulf { pe = (%i1, %i2, %i3, %i4) } %5, %6 : f16
        %9 = arith.addf { pe = (%i1, %i2, %i3, %i4) } %7, %8 : f16
        memref.store %9, %tmp[%i1, %i2, %i3, %i4] 