%3 = memref.load %bias[%c2]
memref.store %3, %tmp[%c1, %c2, %c3, %c4]
...
%40 = memref.load %cnp[%c1, %c0, %38, %39]
%41 = memref.load %weight[%c2, %c0, %c1, %c2]
%42 = memref.load %tmp[%c1, %c2, %c3, %c4]
%43 = arith.mulf { pe = $\setlength{\fboxsep}{1pt}\colorbox{highlightcolor2}{(%c1, %c2, %c3, %c4)}$ } %40, %41
$\setlength{\fboxsep}{1pt}\colorbox{highlightcolor1}{%44}$ = arith.addf { pe = $\setlength{\fboxsep}{1pt}\colorbox{highlightcolor1}{(%c1, %c2, %c3, %c4)}$ } %42, %43
...
%53 = arith.mulf { pe = $\setlength{\fboxsep}{1pt}\colorbox{highlightcolor2}{(%c1, %c2, %c3, %c4)}$ } $\setlength{\fboxsep}{1pt}\colorbox{highlightcolor1}{%44}$, %51
%54 = arith.addf { pe = $\setlength{\fboxsep}{1pt}\colorbox{highlightcolor1}{(%c1, %c2, %c3, %c4)}$ } %52, %53