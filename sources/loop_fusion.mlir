scf.for %arg1 = %c0 to %c1 step %c1 :
  scf.for %arg2 = %c0 to %c64 step %c1 :
    scf.for %arg3 = %c0 to %c9 step %c1 :
      scf.for %arg4 = %c0 to %c9 step %c1 :
        %3 = memref.load %1[%arg2]
        memref.store %3, %2[%arg1, %arg2, %arg3, %arg4]

scf.for %arg1 = %c0 to %c1 step %c1 :
  scf.for %arg2 = %c0 to %c64 step %c1 :
    scf.for %arg3 = %c0 to %c9 step %c1 :
      scf.for %arg4 = %c0 to %c9 step %c1 :
        scf.for %arg5 = %c0 to %c1 step %c1 :
          scf.for %arg6 = %c0 to %c3 step %c1 :
            scf.for %arg7 = %c0 to %c3 step %c1 :
              %3 = arith.addi %arg3, %arg6
              %4 = arith.addi %arg4, %arg7
              %5 = memref.load %arg0[%arg1, %arg5, %3, %4]
              %6 = memref.load %0[%arg2, %arg5, %arg6, %arg7]
              %7 = memref.load %2[%arg1, %arg2, %arg3, %arg4]
              %8 = arith.mulf %5, %6
              %9 = arith.addf %7, %8
              memref.store %9, %2[%arg1, %arg2, %arg3, %arg4]