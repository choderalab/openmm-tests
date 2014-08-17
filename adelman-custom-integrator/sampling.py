#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Sampling utility functions.

DESCRIPTION

COPYRIGHT

@author John D. Chodera <jchodera@gmail.com>

All code in this repository is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

TODO

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import time
import copy

import numpy

import simtk
import simtk.unit as units
import simtk.openmm as openmm
    
#=============================================================================================
# CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA

#=============================================================================================
# Utility functions
#=============================================================================================

def compute_forces(context, positions):
    """
    Compute forces for given positions.
    """

    context.setPositions(positions)
    state = context.getState(getForces=True)
    forces = state.getForces(asNumpy=True)
    return forces

def compute_energy(context, positions, velocities):
    """
    Compute total energy for positions and velocities.
    """
    # Set positions and velocities.
    context.setPositions(positions)
    context.setVelocities(velocities)
    # Compute total energy.
    state = context.getState(getEnergy=True)
    total_energy = state.getPotentialEnergy() + state.getKineticEnergy()

    #print "potential energy: %.3f kcal/mol" % (state.getPotentialEnergy() / units.kilocalories_per_mole)
    #print "kinetic   energy: %.3f kcal/mol" % (state.getKineticEnergy() / units.kilocalories_per_mole)    
    
    return total_energy

def compute_forces_and_energy(context, positions, velocities):
    """
    Compute total energy for positions and velocities.
    """
    # Set positions and velocities.
    context.setPositions(positions)
    context.setVelocities(velocities)
    # Compute total energy.
    state = context.getState(getForces=True, getEnergy=True)
    forces = state.getForces(asNumpy=True)
    total_energy = state.getPotentialEnergy() + state.getKineticEnergy()

    #print "potential energy: %.3f kcal/mol" % (state.getPotentialEnergy() / units.kilocalories_per_mole)
    #print "kinetic   energy: %.3f kcal/mol" % (state.getKineticEnergy() / units.kilocalories_per_mole)    
    
    return [forces, total_energy]

def compute_potential(context, positions):
    """
    Compute potential energy for positions.
    """
    # Set positions and velocities.
    context.setPositions(positions)
    # Compute total energy.
    state = context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy() 
    return potential_energy

def minimize(platform, system, positions):
    # Create a Context.
    timestep = 1.0 * units.femtoseconds
    integrator = openmm.VerletIntegrator(timestep)
    context = openmm.Context(system, integrator, platform)
    # Set coordinates.
    context.setPositions(positions)
    # Compute initial energy.
    state = context.getState(getEnergy=True)
    initial_potential = state.getPotentialEnergy()
    print "initial potential: %12.3f kcal/mol" % (initial_potential / units.kilocalories_per_mole)
    # Minimize.
    openmm.LocalEnergyMinimizer.minimize(context)
    # Compute final energy.
    state = context.getState(getEnergy=True, getPositions=True)
    final_potential = state.getPotentialEnergy()
    positions = state.getPositions(asNumpy=True)
    # Report
    print "final potential  : %12.3f kcal/mol" % (final_potential / units.kilocalories_per_mole)

    return positions

#=============================================================================================
# Equilibrate the system with OpenMM's leapfrog Langevin dynamics.
#=============================================================================================

def equilibrate_langevin(system, timestep, collision_rate, temperature, sqrt_kT_over_m, coordinates, platform):
    nsteps = 5000

    print "Equilibrating for %.3f ps..." % ((nsteps * timestep) / units.picoseconds)
    
    # Create integrator and context.
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(system, integrator, platform)

    # Set coordinates.
    context.setPositions(coordinates)

    # Set Maxwell-Boltzmann velocities
    velocities = sqrt_kT_over_m * numpy.random.standard_normal(size=sqrt_kT_over_m.shape)
    context.setVelocities(velocities)

    # Equilibrate.
    integrator.step(nsteps)

    # Compute energy
    print "Computing energy."
    state = context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()
    print "potential energy: %.3f kcal/mol" % (potential_energy / units.kilocalories_per_mole)

    # Get coordinates.
    state = context.getState(getPositions=True, getVelocities=True)    
    coordinates = state.getPositions(asNumpy=True)
    velocities = state.getVelocities(asNumpy=True)    
    box_vectors = state.getPeriodicBoxVectors()
    system.setDefaultPeriodicBoxVectors(*box_vectors)    

    print "Computing energy again."
    context.setPositions(coordinates)
    context.setVelocities(velocities)        
    state = context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()
    print "potential energy: %.3f kcal/mol" % (potential_energy / units.kilocalories_per_mole)
    
    total_energy = compute_energy(context, coordinates, velocities)    

    return [coordinates, velocities]

#=============================================================================================
# Equilibrate the system with hybrid Monte Carlo (HMC) dynamics.
#=============================================================================================

def equilibrate_hmc(system, timestep, collision_rate, temperature, masses, sqrt_kT_over_m, coordinates, platform, debug=False):

    nhmc = 100 # number of HMC iterations
    nsteps = 50 # number of steps per HMC iteration
    beta = 1.0 / (kB * temperature)

    print "Equilibrating for %d HMC moves of %.3f ps..." % (nhmc, (nsteps * timestep) / units.picoseconds)
    
    # Create integrator and context.
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(system, integrator, platform)

    naccepted = 0
    for hmc_move in range(nhmc):
        if debug: print "HMC move %5d / %5d" % (hmc_move, nhmc)
        
        # Set coordinates.
        context.setPositions(coordinates)
        
        # Set Maxwell-Boltzmann velocities
        velocities = sqrt_kT_over_m * numpy.random.standard_normal(size=sqrt_kT_over_m.shape)
        context.setVelocities(velocities)

        # Compute initial total energy.
        state = context.getState(getEnergy=True)
        initial_energy = state.getPotentialEnergy() + state.getKineticEnergy()

        # Half-kick velocities backwards.
        state = context.getState(getForces=True)
        forces = state.getForces(asNumpy=True)
        velocities[:,:] -= 0.5 * forces[:,:]/masses[:,:] * timestep
        context.setVelocities(velocities)
        
        # Integrate using leapfrog.
        integrator.step(nsteps)

        # Half-kick velocities forwards.
        state = context.getState(getForces=True,getVelocities=True)
        forces = state.getForces(asNumpy=True)
        velocities = state.getVelocities(asNumpy=True)
        velocities[:,:] += 0.5 * forces[:,:]/masses[:,:] * timestep         
        context.setVelocities(velocities)

        # Compute final energy.
        state = context.getState(getEnergy=True)
        final_energy = state.getPotentialEnergy() + state.getKineticEnergy()

        # Compute HMC acceptance.
        du = beta * (final_energy - initial_energy)
        if debug: print "du = %8.1f :" % du,
        if numpy.random.uniform() < numpy.exp(-du):
            # Accept.
            if debug: print " accepted."
            state = context.getState(getPositions=True)
            coordinates = state.getPositions(asNumpy=True)
            naccepted += 1
        else:
            # Reject.
            if debug: print " rejected."
            pass
        
    # Compute energy
    if debug: print "Computing energy."
    state = context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()
    if debug: print "potential energy: %.3f kcal/mol" % (potential_energy / units.kilocalories_per_mole)

    # Get coordinates.
    state = context.getState(getPositions=True, getVelocities=True)    
    coordinates = state.getPositions(asNumpy=True)
    velocities = state.getVelocities(asNumpy=True)    
    box_vectors = state.getPeriodicBoxVectors()
    system.setDefaultPeriodicBoxVectors(*box_vectors)    

    if debug: print "Computing energy again."
    context.setPositions(coordinates)
    context.setVelocities(velocities)        
    state = context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()
    if debug: print "potential energy: %.3f kcal/mol" % (potential_energy / units.kilocalories_per_mole)
    
    total_energy = compute_energy(context, coordinates, velocities)    

    fraction_accepted = float(naccepted) / float(nhmc)

    return [coordinates, velocities, fraction_accepted]

#=============================================================================================
# Generalized hybrid Monte Carlo (GHMC) integrator
#=============================================================================================

def equilibrate_ghmc(system, timestep, collision_rate, temperature, masses, sqrt_kT_over_m, positions, platform, debug=False):

    nsteps = 500 # number of steps
    kT = kB * temperature
    beta = 1.0 / kT # inverse temperature

    print "Equilibrating for %d GHMC steps (%.3f ps)..." % (nsteps, (nsteps * timestep) / units.picoseconds)
    initial_time = time.time()
        
    # Assign Maxwell-Boltzmann velocities
    velocities = sqrt_kT_over_m * numpy.random.standard_normal(size=sqrt_kT_over_m.shape)

    # Compute Langevin velocity modification factors.
    gamma = collision_rate * masses
    sigma2 = (2.0 * kT * gamma)
    alpha_factor = (1.0 - (timestep/4.0)*collision_rate) / (1.0 + (timestep/4.0)*collision_rate)
    x = (timestep/2.0*sigma2).in_units_of(units.kilogram**2/units.mole**2 * units.meter**2/units.second**2)
    y = units.Quantity(numpy.sqrt(x / x.unit), units.sqrt(1.0 * x.unit))
    beta_factor = (y / (1.0 + (timestep/4.0)*collision_rate) / masses).in_units_of(velocities.unit)

    # Create integrator and context.
    integrator = openmm.VerletIntegrator(timestep)
    context = openmm.Context(system, integrator, platform)

    # Compute forces and total energy.
    context.setPositions(positions)
    context.setVelocities(velocities)
    state = context.getState(getForces=True, getEnergy=True)
    forces = state.getForces(asNumpy=True)
    kinetic_energy = state.getKineticEnergy()
    potential_energy = state.getPotentialEnergy()
    total_energy = kinetic_energy + potential_energy

    # Create storage for proposed positions and velocities.
    proposed_positions = copy.deepcopy(positions)
    proposed_velocities = copy.deepcopy(velocities)

    naccepted = 0 # number of accepted GHMC steps
    for step in range(nsteps):
        #
        # Velocity modification step.
        #

        velocities[:,:] = velocities[:,:]*alpha_factor + units.Quantity(numpy.random.standard_normal(size=positions.shape) * (beta_factor/beta_factor.unit), beta_factor.unit)
        kinetic_energy = 0.5 * (masses * velocities**2).in_units_of(potential_energy.unit).sum() * potential_energy.unit # have to do this because sum(...) and .sum() don't respect units
        total_energy = kinetic_energy + potential_energy
        
        #
        # Metropolis-wrapped Velocity Verlet step
        # 

        proposed_positions[:,:] = positions[:,:]
        proposed_velocities[:,:] = velocities[:,:]
                
        # Half-kick velocities
        proposed_velocities[:,:] += 0.5 * forces[:,:]/masses[:,:] * timestep 
        
        # Full-kick positions
        proposed_positions[:,:] += proposed_velocities[:,:] * timestep 

        # Update force at new positions.
        context.setVelocities(proposed_velocities)
        context.setPositions(proposed_positions)
        state = context.getState(getForces=True, getEnergy=True)
        proposed_forces = state.getForces(asNumpy=True)
        proposed_potential_energy = state.getPotentialEnergy()
        
        # Half-kick velocities
        proposed_velocities[:,:] += 0.5 * proposed_forces[:,:]/masses[:,:] * timestep
        proposed_kinetic_energy = 0.5 * (masses * proposed_velocities**2).in_units_of(potential_energy.unit).sum() * potential_energy.unit # have to do this because sum(...) and .sum() don't respect units

        # Compute new total energy.
        proposed_total_energy = proposed_kinetic_energy + proposed_potential_energy
        
        # Accept or reject, inverting momentum if rejected.
        du = beta * (proposed_total_energy - total_energy)
        if (du < 0.0) or (numpy.random.uniform() < numpy.exp(-du)):
            # Accept and update positions, velocities, forces, and energies.
            naccepted += 1
            positions[:,:] = proposed_positions[:,:]
            velocities[:,:] = proposed_velocities[:,:]        
            forces[:,:] = proposed_forces[:,:]
            potential_energy = proposed_potential_energy
            kinetic_energy = proposed_kinetic_energy
        else:
            # Reject, requiring negation of velocities.
            velocities[:,:] = -velocities[:,:]        

        #
        # Velocity modification step.
        #

        velocities[:,:] = velocities[:,:]*alpha_factor + units.Quantity(numpy.random.standard_normal(size=positions.shape) * (beta_factor/beta_factor.unit), beta_factor.unit)
        kinetic_energy = 0.5 * (masses * velocities**2).in_units_of(potential_energy.unit).sum() * potential_energy.unit # have to do this because sum(...) and .sum() don't respect units
        total_energy = kinetic_energy + potential_energy
        
    # Print final statistics.
    fraction_accepted = float(naccepted) / float(nsteps)
    final_time = time.time()
    elapsed_time = final_time - initial_time
    if debug: print "%12.3f s elapsed | accepted %6.3f%%" % (elapsed_time, fraction_accepted*100.0)
    
    return [positions, velocities, fraction_accepted]
