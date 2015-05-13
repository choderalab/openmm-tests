#!/usr/bin/env python

"""
Energy continuity test for two TIP3P molecules.

"""

import simtk.openmm as mm
from simtk.openmm import app
from simtk import unit
import numpy as np
import pandas as pd

# Parameters
switch_width = 1.0 * unit.angstrom
cutoff = 9.0 * unit.angstrom
use_switch = True
box = 2*cutoff + 1.0 * unit.angstrom
nonbondedMethod = app.CutoffPeriodic

# Read PDB file.
pdb_filename = 'water-dimer.pdb'
pdbfile = app.PDBFile(pdb_filename)

# Construct system.
forcefields_to_use = ['amber99sbildn.xml', 'tip3p.xml'] # list of forcefields to use in parameterization
forcefield = app.ForceField(*forcefields_to_use)
system = forcefield.createSystem(pdbfile.topology, nonbondedMethod=nonbondedMethod, constraints=app.HBonds)

# Get positions.
positions = pdbfile.getPositions()

# Change box size.
system.setDefaultPeriodicBoxVectors((box, 0, 0), (0, box, 0), (0, 0, box))

# Move particles.
positions = unit.Quantity(np.array(positions / unit.angstroms), unit.angstroms)
# Center molecules.
positions[0:2,:] -= positions[0,:]
positions[3:5,:] -= positions[3,:]
# Offset molecules.
positions[3:5,1] += 3.0 * unit.angstrom

# velocities
velocities = unit.Quantity(np.zeros([6,3]), unit.nanometers/unit.picoseconds)
velocities[0:2,0] = 1.0 * unit.nanometers/unit.picoseconds

# Modify NonbondedForce.
forces = { system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces()) }
f = forces['NonbondedForce']
f.setCutoffDistance(cutoff)
f.setSwitchingDistance(cutoff - switch_width)
f.setUseDispersionCorrection(False)
f.setUseSwitchingFunction(use_switch)

xmax = 4*box
n = 10000
dt = (xmax) / n
integrator = mm.CustomIntegrator(dt)

integrator.addPerDofVariable("xcur", 0.0)
integrator.addPerDofVariable("fcur", 0.0)
integrator.addPerDofVariable("fsum", 0.0)
integrator.addGlobalVariable("steps", 0.0)
integrator.addGlobalVariable("ecur", 0.0)


integrator.addComputePerDof("x", "x + dt * v")
integrator.addComputePerDof("xcur", "x")
integrator.addComputePerDof("fsum", "fsum + f")
integrator.addComputePerDof("fcur", "f")
integrator.addComputeGlobal("steps", "steps + 1")
integrator.addComputeGlobal("ecur", "energy")

context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocities(velocities)

data = []
for i in range(n):
    integrator.step(1)
    xcur = integrator.getPerDofVariableByName("xcur")
    fsum = integrator.getPerDofVariableByName("fsum")
    fcur = integrator.getPerDofVariableByName("fcur")
    steps = integrator.getGlobalVariableByName("steps")
    ecur = integrator.getGlobalVariableByName("ecur")
    di = dict(steps=steps, xcur=xcur[1][0], fsum=fsum[1][0], fcur=fcur[1][0], ecur=ecur)
    print(di)
    data.append(di)

data = pd.DataFrame(data)

from pylab import *
from matplotlib.backends.backend_pdf import PdfPages

#with PdfPages('out.pdf') as pdf:
plot(data.xcur, data.ecur, linewidth=1)
xlabel("position (nm)")
ylabel("energy (kJ/mol)")
oldaxis = axis()
axis([0, xmax / unit.nanometers, oldaxis[2], oldaxis[3]])
#pdf.savefig()
savefig('potential.png', bbox_inches='tight')

figure()
plot(data.xcur, data.fcur, linewidth=1)
xlabel("position (nm)")
ylabel("force (1 component) (kJ/mol/nm)")
oldaxis = axis()
axis([0, xmax / unit.nanometers, oldaxis[2], oldaxis[3]])
#pdf.savefig()
savefig('force.png', bbox_inches='tight')
