#!/bin/bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$PWD/helios-plusplus-lin/run"
export LD_LIBRARY_PATH

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

python panel_helios_sim.py