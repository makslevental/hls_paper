scf.for %i1 = %c0 to %c1 step %c1 :
  scf.for %i2 = %c0 to %c64 step %c1 :
    scf.for %i3 = %c0 to %c9 step %c1 :
      scf.for %i4 = %c0 to %c9 step %c1 :
        scf.for %i5 = %c0 to %c1 step %c1 :
          scf.for %i6 = %c0 to %c3 step %c1 :
            scf.for %i7 = %c0 to %c3 step %c1 :
              ...
              memref.store %9, %tmp[%i1, %i2, %i3, %i4] 

...            

scf.for %i1 = %c0 to %c1 step %c1 :
  scf.for %i2 = %c0 to %c32 step %c1 :
    scf.for %i3 = %c0 to %c9 step %c1 :
      scf.for %i4 = %c0 to %c9 step %c1 :
        scf.for %i5 = %c0 to %c64 step %c1 :
          scf.for %i6 = %c0 to %c1 step %c1 :
            scf.for %i7 = %c0 to %c1 step %c1 :
              ...
              %5 = memref.load %tmp[%i1, %i5, %3, %4] 