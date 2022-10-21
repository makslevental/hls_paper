source /opt/tools/xilinx/Vitis_HLS/2022.1/settings64.sh

nets=(
#addmm
#batch_norm
#conv
#max_pool_2d
#soft_max
braggnn
)
  
unroll_factors=(
1
4
8
12
16
20
24
28
32
36
40
1024
)

for net in "${nets[@]}"; do
  cp run_hls.tcl ${net}_16
  pushd ${net}_16
  for unroll_factor in "${unroll_factors[@]}"; do
    while [ $(jobs | wc -l) -ge 4 ] ; do sleep 1 ; done
    rm -rf ${net}_${unroll_factor}
    MODULE=${net}_${unroll_factor} vitis_hls run_hls.tcl 
  done
  popd
done
