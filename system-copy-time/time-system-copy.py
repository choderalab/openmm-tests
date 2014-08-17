# Time system copy operations.

import copy
import time

import simtk.openmm as mm
from repex import testsystems

names = ['SrcImplicit', 'SrcExplicit', 'LysozymeImplicit', 'AlchemicalLennardJonesCluster']

for name in names:
    constructor = getattr(testsystems, name)
    testsystem = constructor()
    system = testsystem.system
    print "%s (%d atoms)" % (name, system.getNumParticles())

    initial_time = time.time()
    system_copy = copy.deepcopy(system)
    final_time = time.time()
    elapsed_time = final_time - initial_time
    print "Copying took %.3f seconds" % elapsed_time

    print ""
