import simtk.openmm as mm
from simtk import unit
import pandas as pd

sigma = 3.4 * unit.angstrom # argon
epsilon = 0.238 * unit.kilocalories_per_mole # argon

#switch_width = 0.5 * unit.angstrom
switch_width = 1.0 * sigma
cutoff = 3.0 * sigma
use_switch = True
box = 2*cutoff + 1.0 * unit.angstrom

system = mm.System()
system.setDefaultPeriodicBoxVectors((box, 0, 0), (0, box, 0), (0, 0, box))
system.addParticle(1.0)
system.addParticle(1.0)

f = mm.NonbondedForce()
f.setCutoffDistance(cutoff)
f.setSwitchingDistance(cutoff - switch_width)
f.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)

f.setUseDispersionCorrection(False)
f.setUseSwitchingFunction(use_switch)


f.addParticle(0.0, sigma, epsilon)
f.addParticle(0.0, sigma, epsilon)
system.addForce(f)

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
context.setVelocities([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
context.setPositions([(0.0, 0.0, 0.0), (0., 1.5*sigma, 0.0)])

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
