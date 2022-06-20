scf.for %i1 = %c0 to %c1 step %c1 :
  scf.for %i2 = %c0 to %c64 step %c1 :
    scf.for %i3 = %c0 to %c9 step %c1 :
      scf.for %i4 = %c0 to %c9 step %c1 :
        %3 = memref.load %bias[%i2]
        memref.store %3, %tmp[%i1, %i2, %i3, %i4]
        ...
        %c2 = arith.constant 2
        %38 = arith.addi %i3, %c1
        %39 = arith.addi %i4, %c2
        %40 = memref.load %inp[%i1, %c0, %38, %39]
        %41 = memref.load %weight[%i2, %c0, %c1, %c2]
        %42 = memref.load %tmp[%i1, %i2, %i3, %i4]
        %43 = arith.mulf %40, %41
        %44 = arith.addf %42, %43
        memref.store %44, %tmp[%i1, %i2, %i3, %i4]
        ...