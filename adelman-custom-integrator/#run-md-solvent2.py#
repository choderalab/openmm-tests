

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Simulation of WCA dimer in dense WCA solvent using GHMC.

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

import os
import os.path
import sys
import math
import copy
import time

import numpy

import simtk
import simtk.unit as units
import simtk.openmm as openmm
    
#import Scientific.IO.NetCDF as netcdf # for netcdf interface in Scientific
import netCDF4 as netcdf # for netcdf interface provided by netCDF4 in enthought python

import wcadimer
import sampling

from integrators import GHMCIntegrator

#=============================================================================================
# SUBROUTINES
#=============================================================================================

def norm(n01):
    return n01.unit * numpy.sqrt(numpy.dot(n01/n01.unit, n01/n01.unit))

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    # PARAMETERS
    netcdf_filename = 'data/md-solvent.nc'

    verbose = False
    
    # WCA fluid parameters (argon).
    mass     = wcadimer.mass
    sigma    = wcadimer.sigma
    epsilon  = wcadimer.epsilon
    r_WCA    = wcadimer.r_WCA
    r0       = wcadimer.r0
    h        = wcadimer.h
    w        = wcadimer.w
    
    # Compute characteristic timescale.
    tau = wcadimer.tau
    print "tau = %.3f ps" % (tau / units.picoseconds)

    # Compute timestep.
    equilibrate_timestep = 2 * wcadimer.stable_timestep
    timestep = 5 * wcadimer.stable_timestep
    #timestep = equilibrate_timestep # DEBUG
    print "equilibrate timestep = %.1f fs, switch timestep = %.1f fs" % (equilibrate_timestep / units.femtoseconds, timestep / units.femtoseconds)

    # Set temperature, pressure, and collision rate for stochastic thermostats.
    temperature = wcadimer.temperature
    print "temperature = %.1f K" % (temperature / units.kelvin)
    kT = wcadimer.kT
    beta = 1.0 / kT # inverse temperature    
    collision_rate = 1.0 / tau # collision rate for Langevin integrator
    print 'collision_rate: ', collision_rate

    niterations = 10 # number of work samples to collect

    # Create system.     
    [system, coordinates] = wcadimer.WCADimer()

    # Form vectors of masses and sqrt(kT/m) for force propagation and velocity randomization.
    print "Creating masses array..."
    nparticles = system.getNumParticles()
    masses = units.Quantity(numpy.zeros([nparticles,3], numpy.float64), units.amu)
    for particle_index in range(nparticles):
        masses[particle_index,:] = system.getParticleMass(particle_index)
    sqrt_kT_over_m = units.Quantity(numpy.zeros([nparticles,3], numpy.float64), units.nanometers / units.picosecond)
    for particle_index in range(nparticles):
        sqrt_kT_over_m[particle_index,:] = units.sqrt(kT / masses[particle_index,0]) # standard deviation of velocity distribution for each coordinate for this atom

    # List all available platforms
    print "Available platforms:"
    for platform_index in range(openmm.Platform.getNumPlatforms()):
        platform = openmm.Platform.getPlatform(platform_index)
        print "%5d %s" % (platform_index, platform.getName())
    print ""

    # Select platform.
    #platform = openmm.Platform.getPlatformByName("CPU")
    #platform = openmm.Platform.getPlatformByName("CPU")
    platform = openmm.Platform.getPlatformByName("OpenCL")
    #platform = openmm.Platform.getPlatformByName("CUDA")
    #min_platform = openmm.Platform.getPlatformByName("Reference")
    min_platform = openmm.Platform.getPlatformByName("CPU")
    #deviceid = 2
    #platform.setPropertyDefaultValue('OpenCLDeviceIndex', '%d' % deviceid)
    #platform.setPropertyDefaultValue('CudaDeviceIndex', '%d' % deviceid)         
    platform.setPropertyDefaultValue('OpenCLPrecision', 'double')
    #platform.setPropertyDefaultValue('CudaPrecision', 'double')

    # Initialize netcdf file.
    if not os.path.exists(netcdf_filename):
        # Open NetCDF file for writing
        ncfile = netcdf.Dataset(netcdf_filename, 'w') # for netCDF4
        
        # Create dimensions.
        ncfile.createDimension('iteration', 0) # unlimited number of iterations
        ncfile.createDimension('nparticles', nparticles) # number of particles
        ncfile.createDimension('ndim', 3) # number of dimensions    

        # Create variables.
        ncfile.createVariable('distance', 'd', ('iteration',))
        ncfile.createVariable('positions', 'd', ('iteration','nparticles','ndim'))        
        ncfile.createVariable('fraction_accepted', 'd', ('iteration',))
        
        # Force sync to disk to avoid data loss.
        ncfile.sync()

        # Minimize.
        print "Minimizing energy..."
        coordinates = sampling.minimize(min_platform, system, coordinates)
    
        # Equilibrate.
        print "Equilibrating..."
        [coordinates, velocities, fraction_accepted] = sampling.equilibrate_ghmc(system, equilibrate_timestep, collision_rate, temperature, masses, sqrt_kT_over_m, coordinates, platform)
        print "%.3f %% accepted" % (fraction_accepted * 100.0)
        
        # Write initial configuration.
        ncfile.variables['distance'][0] = norm(coordinates[1,:] - coordinates[0,:]) / units.angstroms
        ncfile.variables['positions'][0,:,:] = coordinates[:,:] / units.angstroms
        ncfile.sync()        
        iteration = 1
    else:
        # Open NetCDF file for reading.
        ncfile = netcdf.Dataset(netcdf_filename, 'a') # for netCDF4

        # Read iteration and coordinates.
        iteration = ncfile.variables['distance'][:].size
        coordinates = units.Quantity(ncfile.variables['positions'][iteration-1,:,:], units.angstroms)

    # Continue
    print timestep.in_units_of(units.femtosecond)
    print temperature.in_units_of(units.kelvin)
    print collision_rate.in_units_of(units.picosecond**-1)
    ghmc_integrator = GHMCIntegrator(timestep=timestep, temperature=temperature,
                                     collision_rate=collision_rate)
    ghmc_global_variables = { ghmc_integrator.getGlobalVariableName(index) : index for index in range(ghmc_integrator.getNumGlobalVariables()) }

    context = openmm.Context(system, ghmc_integrator, platform)
    context.setPositions(coordinates)

    state = context.getState(getPositions=True)

    while (iteration < niterations):
        print "iteration %5d / %5d" % (iteration, niterations)
        print 'coordinates: ', coordinates[0,:] / units.angstrom
        initial_time = time.time()
        
        # Generate a new configuration.
        initial_distance = norm(coordinates[1,:] - coordinates[0,:])
        ghmc_integrator.setGlobalVariable(ghmc_global_variables['naccept'], 0)
        ghmc_integrator.setGlobalVariable(ghmc_global_variables['ntrials'], 0)

        ghmc_integrator.step(500)

        naccept = ghmc_integrator.getGlobalVariable(ghmc_global_variables['naccept'])
        ntrials = ghmc_integrator.getGlobalVariable(ghmc_global_variables['ntrials'])
        fraction_accepted = float(naccept) / float(ntrials)
        
        print "%.3f %% accepted" % (fraction_accepted * 100.0)
        
        state = context.getState(getPositions=True)
        coordinates = state.getPositions(asNumpy=True)
        
        final_distance = norm(coordinates[1,:] - coordinates[0,:])            
        print "Dynamics %.1f A -> %.1f A (barrier at %.1f A)" % (initial_distance / units.angstroms, final_distance / units.angstroms, (r0+w)/units.angstroms)
        ncfile.variables['fraction_accepted'][iteration] = fraction_accepted
        
        # Record attempt
        ncfile.variables['distance'][iteration] = final_distance / units.angstroms
        ncfile.variables['positions'][iteration,:,:] = coordinates[:,:] / units.angstroms
        ncfile.sync()

        # Debug.
        final_time = time.time()
        elapsed_time = final_time - initial_time
        print "%12.3f s elapsed" % elapsed_time

        # Increment iteration counter.
        iteration += 1

    # Close netcdf file.
    ncfile.close()
