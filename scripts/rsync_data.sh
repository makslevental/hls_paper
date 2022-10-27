rsync -zmarv vastvm1:~/fully_unrolled_cpps/reports .
rsync -zmarv vastvm2:~/fully_unrolled_cpps/reports .

#rsync -zmarv --include="*/" --include="*.rpt" --exclude=".*/" --exclude="*" wing-artemis:~/bragghls_artifacts .
#rsync -zmarv --include="*/" --include="*.sched.mlir" --exclude=".*/" --exclude="*" wing-artemis:~/bragghls_artifacts .
