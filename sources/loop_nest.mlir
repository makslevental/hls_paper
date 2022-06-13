scf.for %arg1 = %c0 to %c1 step %c1 :
  scf.for %arg2 = %c0 to %c64 step %c1 :
    scf.for %arg3 = %c0 to %c9 step %c1 :
      scf.for %arg4 = %c0 to %c9 step %c1 :
        %3 = memref.load %1[%arg2] : memref
        memref.store %3, %2[%arg1, %arg2, %arg3, %arg4] : memref

scf.for %arg1 = %c0 to %c1 step %c1 :
  scf.for %arg2 = %c0 to %c64 step %c1 :
    scf.for %arg3 = %c0 to %c9 step %c1 :
      scf.for %arg4 = %c0 to %c9 step %c1 :
        scf.for %arg5 = %c0 to %c1 step %c1 :
          scf.for %arg6 = %c0 to %c3 step %c1 :
            scf.for %arg7 = %c0 to %c3 step %c1 :
              %3 = arith.addi %arg3, %arg6 : i32
              %4 = arith.addi %arg4, %arg7 : i32
              %5 = memref.load %arg0[%arg1, %arg5, %3, %4] : memref
              %6 = memref.load %0[%arg2, %arg5, %arg6, %arg7] : memref
              %7 = memref.load %2[%arg1, %arg2, %arg3, %arg4] : memref
              %8 = arith.mulf %5, %6 : f16
              %9 = arith.addf %7, %8 : f16
              memref.store %9, %2[%arg1, %arg2, %arg3, %arg4] : memref