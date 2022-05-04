scf.for %arg1 = %c0 to %c1 step %c1 :
  scf.for %arg2 = %c0 to %c64 step %c1 :
    scf.for %arg3 = %c0 to %c9 step %c1 :
      scf.for %arg4 = %c0 to %c9 step %c1 :
        %3 = memref.load %1[%arg2]
        memref.store %3, %2[%arg1, %arg2, %arg3, %arg4]
        ...
        %c2 = arith.constant 2
        %38 = arith.addi %arg3, %c1
        %39 = arith.addi %arg4, %c2
        %40 = memref.load %arg0[%arg1, %c0, %38, %39]
        %41 = memref.load %0[%arg2, %c0, %c1, %c2]
        %42 = memref.load %2[%arg1, %arg2, %arg3, %arg4]
        %43 = arith.mulf %40, %41
        %44 = arith.addf %42, %43
        memref.store %44, %2[%arg1, %arg2, %arg3, %arg4]
        ...