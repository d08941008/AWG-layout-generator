import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from AWGgenerator import *
os.chdir(os.path.dirname(os.path.realpath(__file__)))

##......................................................... Si main ..............................................................#
indFName = '.\\AWG_Si_20220331.ind'
onlyParam, narFileOverwrite = 0, 0
Lambda0, DLambda = 1.3, 0.02
Width, Width_fat = 0.38, 0.68
Nin, Nout, Nchan, M = 3, 6, 6, 16
Di, Do, Wit, Wot = 1.9, 1.9, 1.7, 1.7
arc_radius = 10
arcend_pitch, DL_pitch, port_pitch = 2.6, 4, 3
shape = 2                          # 0 for smit; 1 for rectangular; 2 for S-shaped;
naFName = 'Neff_Si_test20220328.dat'        # test for silicon-based
narFName = 'NeffR_Si_test20220328.dat'      # test fot silicon-based
core_material = 'Si_material.dat'
clad_material = 'SiO2_material.dat'
height = 0.22

inputCenter = (1000, 100)   # interface center between input port and star coupler, to avoid the GDS-II export bug
## ...............................................................................................................................................................................##

AWG_rsoftInd_generate(indFName, naFName, narFName, core_material, clad_material, Nin, Nout, Nchan, M, shape, 
                                                Di, Do, Lambda0, DLambda, Wit, Wot, Width, Width_fat, arc_radius, inputCenter, 
                                                DL_pitch, arcend_pitch, port_pitch, onlyParam = 0, narFileOverwrite = 0, height = height)