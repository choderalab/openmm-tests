#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Estimate RMS fluctuations of total energy for OpenMM as a function of various properties.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import sys
import math
import doctest
import time
import numpy
import os
import os.path

import simtk.unit as units
import simtk.openmm as openmm

from openmmtools import testsystems

import netCDF4 as netcdf

#=============================================================================================
# CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA

#=============================================================================================
# UTILITY SUBROUTINES
#=============================================================================================

def initialize_equilibrated_netcdf(ncfile, system):
    """
    Initialize NetCDF file for storage.
    
    """    
    
    # Create dimensions.
    ncfile.createDimension('atoms', system.getNumParticles()) # number of particles in system
    ncfile.createDimension('dimensions', 3) # number of spatial dimensions
    ncfile.createDimension('single', 1)

    # Create variables.
    ncvar_positions = ncfile.createVariable('positions', 'f8', ('atoms', 'dimensions'))
    ncvar_velocities = ncfile.createVariable('velocities', 'f8', ('atoms', 'dimensions'))
    ncvar_box_vectors = ncfile.createVariable('box_vectors', 'f8', ('dimensions','dimensions'))        

    # Serialize OpenMM System object.
    ncvar_serialized_state = ncfile.createVariable('system', str, ('single',), zlib=True)
    ncvar_serialized_state[0] = system.__getstate__()
    
    # Define units for variables.
    setattr(ncvar_positions, 'units', 'nm')
    setattr(ncvar_velocities, 'units', 'nm/ps')
    setattr(ncvar_box_vectors, 'units', 'nm')

    # Define long (human-readable) names for variables.
    setattr(ncvar_box_vectors, "long_name", "box_vectors[sample,i,j] is dimension j of box vector i for sample 'sample'.")
    setattr(ncvar_positions, "long_name", "positions[sample,particle,dimenson] is position of coordinate 'dimension' of particle 'particle' for sample 'sample'.")
    setattr(ncvar_velocities, "long_name", "velocities[sample,particle,dimension] is velocity of coordinate 'dimension' of particle 'particle' for sample 'sample.")
    
    # Force sync to disk to avoid data loss.
    ncfile.sync()
        
    return

def initialize_netcdf(ncfile, system, ntimesteps_to_try, nrecords):
    """
    Initialize NetCDF file for storage.
    
    """    
    
    # Create dimensions.
    ncfile.createDimension('samples', 0) # unlimited number of samples
    ncfile.createDimension('atoms', system.getNumParticles()) # number of particles in system
    ncfile.createDimension('dimensions', 3) # number of spatial dimensions
    ncfile.createDimension('timesteps', ntimesteps_to_try) # number of timesteps to try
    ncfile.createDimension('records', nrecords) # number of timesteps to try
    ncfile.createDimension('single', 1)

    # Create variables.
    #ncvar_positions = ncfile.createVariable('positions', 'f', ('samples', 'atoms', 'dimensions'))
    #ncvar_velocities = ncfile.createVariable('velocities', 'f', ('samples', 'atoms', 'dimensions'))
    #ncvar_box_vectors = ncfile.createVariable('box_vectors', 'f', ('samples','dimensions','dimensions'))        
    #ncvar_volumes  = ncfile.createVariable('volumes', 'f', ('samples',))
    ncvar_total_energy = ncfile.createVariable('total_energy', 'f', ('timesteps','records'))
    ncvar_rms_total_energy = ncfile.createVariable('rms_total_energy', 'f', ('timesteps',))

    # Serialize OpenMM System object.
    #ncvar_serialized_state = ncfile.createVariable('system', str, ('single',), zlib=True)
    #ncvar_serialized_state[0] = system.__getstate__()
    
    # Define units for variables.
    #setattr(ncvar_positions, 'units', 'nm')
    #setattr(ncvar_velocities, 'units', 'nm/ps')
    #setattr(ncvar_box_vectors, 'units', 'nm')
    #setattr(ncvar_volumes, 'units', 'nm**3')
    setattr(ncvar_total_energy, 'units', 'kT')
    setattr(ncvar_rms_total_energy, 'units', 'kT')

    # Define long (human-readable) names for variables.
    #setattr(ncvar_box_vectors, "long_name", "box_vectors[sample,i,j] is dimension j of box vector i for sample 'sample'.")
    #setattr(ncvar_positions, "long_name", "positions[sample,particle,dimenson] is position of coordinate 'dimension' of particle 'particle' for sample 'sample'.")
    #setattr(ncvar_velocities, "long_name", "velocities[sample,particle,dimension] is velocity of coordinate 'dimension' of particle 'particle' for sample 'sample.")
    #setattr(ncvar_volumes, "long_name", "volume[sample] is the box volume for sample 'sample'.")
    
    # Force sync to disk to avoid data loss.
    ncfile.sync()
        
    return

def select_options(options_list, index):
    selected_options = list()
    for option in options_list:
        noptions = len(option)
        selected_options.append(option[index % noptions])
        index = int(index/noptions)
    return selected_options

def get_all_subclasses(cls):
    """
    Return all subclasses of a specified class.

    Parameters
    ----------
    cls : class
       The class for which all subclasses are to be returned.

    Returns
    -------
    all_subclasses : list of class
       List of all subclasses of `cls`.

    """

    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses

def set_pme_tolerance(system, pme_tolerance):
    """
    Set the PME tolerance

    Parameters
    ----------
    system : simtk.openmm.System
        System to be modified
    pme_tolerance : float
        PME tolerance to use

    """
    forces = { system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces()) }
    if 'NonbondedForce' in forces:
        force = forces['NonbondedForce']
        force.setEwaldErrorTolerance(pme_tolerance)
    return

#=============================================================================================
# PARAMETERS
#=============================================================================================

# Sets of parameters to regress over.
systems_to_try = [ cls.__name__ for cls in get_all_subclasses(testsystems.TestSystem) ] # all testsystems
integrators_to_try = ['VerletIntegrator', 'VelocityVerletIntegrator'] # testsystems to try
switching_to_try = [False, True] # switching function flags
platform_names_to_try = ['CUDA', 'OpenCL', 'CPU', 'Reference'] # platform names to try
precision_models_to_try = ['single', 'mixed', 'double'] # precision models to try
constraint_tolerances_to_try = [1.0e-10, 1.0e-5] # constraint tolerances to try (for systems with constraints)
pme_tolerances_to_try = [5.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10]

# Timesteps to try for each parameter set.
timesteps_to_try = units.Quantity([0.0625, 0.125, 0.250, 0.5, 1.0], units.femtoseconds) # MD timesteps to test for each system
ntimesteps_to_try = len(timesteps_to_try)

# Number of GPUs.
#ngpus = 4

# Other data
simulation_length = 100.0 * units.femtoseconds # length of simulation segment
record_interval = 1.0 * units.femtoseconds # time between recording data
nrecords = int(simulation_length / record_interval)

temperature = 298.0 * units.kelvin
pressure = 1.0 * units.atmosphere # pressure for equilibration

ghmc_nsteps = 1000 # number of steps to generate new uncorrelated sample with GHMC
ghmc_timestep = 0.50 * units.femtoseconds
nequil = 100 # number of NPT equilibration iterations

# DEBUG
systems_to_try = [ cls.__name__ for cls in get_all_subclasses(testsystems.TestSystem) if (('Water' in cls.__name__) and ('Giant' not in cls.__name__)) ] # all non-giant water boxes
precision_models_to_try = ['double'] # precision models to try
platform_names_to_try = ['CUDA'] # platform names to try
precision_models_to_try = ['double', 'mixed']
constraint_tolerances_to_try = [1.0e-10]
integrators_to_try = ['VelocityVerletIntegrator']
#nequil = 5 # number of NPT equilibration iterations

verbose = True

kT = kB * temperature # thermal energy
beta = 1.0 / kT # inverse temperature

options_list = [systems_to_try, integrators_to_try, switching_to_try, platform_names_to_try, precision_models_to_try, constraint_tolerances_to_try, pme_tolerances_to_try]
print "nrecords = %d" % nrecords

#=============================================================================================
# Initialize MPI.
#=============================================================================================

try:
    from mpi4py import MPI # MPI wrapper
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    print "Node %d / %d" % (MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size)
except:
    print "mpi4py not available---using serial execution."
    rank = 0
    size = 1

print "%d / %d" % (rank, size)

try:
    MPI.COMM_WORLD.barrier()
except:
    pass

# Count total number of optionsets.
noptionsets = 1
for optionset in options_list:
    noptionsets *= len(optionset)
if rank == 0: print "There are %d option sets to try." % noptionsets

if rank == 0:
    if not os.path.exists('data'):
        os.makedirs('data')
    
#=============================================================================================
# PARALLELIZE EQUILIBRATION
#=============================================================================================

#=============================================================================================
# Equilibrate with GHMC using a barostat.
#=============================================================================================

nsystems = len(systems_to_try) * len(switching_to_try)
for index in range(rank, nsystems, size):

    # Determine system to equilibrate.
    [system_name, switching_flag] = select_options([systems_to_try, switching_to_try], index)

    # Determine NetCDF filename for equilibrated system.
    netcdf_filename = 'data/system-%s-%s.nc' % (system_name, switching_flag)

    # Attempt to resume if file exists.
    if not os.path.exists(netcdf_filename):
        # Select platform.
        platform_name = 'CUDA'
        precision_model = 'double'
        pme_tolerance = 1.0e-6
    
        platform = openmm.Platform.getPlatformByName(platform_name)
        #deviceid = rank % ngpus
        deviceid = 0
        if platform_name == 'CUDA':
            #platform.setPropertyDefaultValue('CudaDeviceIndex', '%d' % deviceid) # select Cuda device index    
            platform.setPropertyDefaultValue('CudaPrecision', precision_model)        
        elif platform_name == 'OpenCL':
            #platform.setPropertyDefaultValue('OpenCLDeviceIndex', '%d' % deviceid) # select OpenCL device index
            platform.setPropertyDefaultValue('OpenCLPrecision', precision_model)        
        print "node %3d using GPU %d platform %s precision %s" % (rank, deviceid, platform_name, precision_model)
        
        # Create system to simulate.
        print ""
        print "Creating system %s..." % system_name
        constructor = getattr(testsystems, system_name)
        import inspect
        if 'switch' in inspect.getargspec(constructor.__init__).args:
            testsystem = constructor(switch=switching_flag)
        else:
            testsystem = constructor()
        [system, positions] = [testsystem.system, testsystem.positions]

        # Set PME tolerance
        set_pme_tolerance(system, pme_tolerance)

        # Determine number of degrees of freedom.
        nvsites = sum([system.isVirtualSite(index) for index in range(system.getNumParticles())])
        nparticles = system.getNumParticles()
        nconstraints = system.getNumConstraints()
        ndof = 3*(nparticles - nvsites) - nconstraints

        nparticles = system.getNumParticles()        
        print "Node %d: Box has %d particles" % (rank, nparticles)

        # Equilibrate with Monte Carlo barostat.
        from openmmtools.integrators import GHMCIntegrator
        import copy
        system_with_barostat = copy.deepcopy(system)
        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        #system_with_barostat.addForce(barostat)
        ghmc_integrator = GHMCIntegrator(timestep=ghmc_timestep, temperature=temperature)
        ghmc_global_variables = { ghmc_integrator.getGlobalVariableName(index) : index for index in range(ghmc_integrator.getNumGlobalVariables()) }

        # Set constraint tolerance.
        constraint_tolerance = 1.0e-10 # tight tolerance
        ghmc_integrator.setConstraintTolerance(constraint_tolerance)

        # Create context.
        ghmc_context = openmm.Context(system_with_barostat, ghmc_integrator, platform)

        # Modify random number seeds to be unique.
        seed = ghmc_integrator.getRandomNumberSeed()
        ghmc_integrator.setRandomNumberSeed(seed + rank)
        barostat.setRandomNumberSeed(seed + rank + size)
            
        # Set positions and velocities.
        ghmc_context.setPositions(positions)

        # Minimize.
        print "node %3d minimizing..." % rank
        openmm.LocalEnergyMinimizer.minimize(ghmc_context, 0.01 * units.kilocalories_per_mole / units.nanometer, 20)
        print "node %3d minimization complete." % rank

        # Compute initial volume.
        state = ghmc_context.getState()
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        volume = box_vectors[0,0] * box_vectors[1,1] * box_vectors[2,2]
        print "node %d: initial volume %8.3f nm^3" % (rank, volume / units.nanometers**3)

        # Equilibrate system with NPT.
        volume_history = numpy.zeros([nequil], numpy.float64)
        for iteration in range(nequil):
            ghmc_integrator.setGlobalVariable(ghmc_global_variables['naccept'], 0)
            ghmc_integrator.setGlobalVariable(ghmc_global_variables['ntrials'], 0)

            # Generate new sample from equilibrium distribution with GHMC.
            ghmc_integrator.step(ghmc_nsteps)
    
            # Compute volume.
            state = ghmc_context.getState(getEnergy=True)
            box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
            potential = state.getPotentialEnergy()
            kinetic = state.getKineticEnergy()
            volume = box_vectors[0,0] * box_vectors[1,1] * box_vectors[2,2]
            volume_history[iteration] = volume / units.nanometers**3
            max_radius = box_vectors[0,0] / 2.0 # half the box width
            instantaneous_kinetic_temperature = kinetic / (ndof * kB / 2.0)

            naccept = ghmc_integrator.getGlobalVariable(ghmc_global_variables['naccept'])
            ntrials = ghmc_integrator.getGlobalVariable(ghmc_global_variables['ntrials'])
            fraction_accepted = float(naccept) / float(ntrials)
            print "%64s : GHMC equil %5d / %5d | accepted %6d / %6d (%7.3f %%) | volume %8.3f nm^3 | max radius %8.3f nm | potential %12.3f kT | temperature %8.3f K" % (system_name, iteration, nequil, naccept, ntrials, fraction_accepted*100.0, volume / units.nanometers**3, max_radius / units.nanometers, potential / kT, instantaneous_kinetic_temperature / units.kelvin)
        
        # Extract coordinates and box vectors.
        state = ghmc_context.getState(getPositions=True, getVelocities=True)
        positions = state.getPositions(asNumpy=True)
        velocities = state.getVelocities(asNumpy=True)
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        volume = box_vectors[0,0] * box_vectors[1,1] * box_vectors[2,2]

        del ghmc_context, ghmc_integrator
        
        # Store equilibrated sample.
        ncfile = netcdf.Dataset(netcdf_filename, 'w', version='NETCDF4')
        initialize_equilibrated_netcdf(ncfile, system)
        ncfile.variables['positions'][:,:] = positions / units.nanometers
        ncfile.variables['velocities'][:,:] = velocities / (units.nanometers / units.picoseconds)
        ncfile.variables['box_vectors'][:,:] = box_vectors / units.nanometers
        ncfile.close()


#=============================================================================================
# Wait for everyone to catch up
#=============================================================================================

try:
    MPI.COMM_WORLD.barrier()
except:
    pass

#=============================================================================================
# Parallelize computing total energy RMS and drift over many combinations of parameters
#=============================================================================================

for index in range(rank, noptionsets, size):

    #=============================================================================================
    # Select problem to work on.
    #=============================================================================================

    try:
        [system_name, integrator_name, switching_flag, platform_name, precision_model, constraint_tolerance, pme_tolerance] = select_options(options_list, index)
    except:
        continue

    #=============================================================================================
    # Create filename to store data in.
    #=============================================================================================

    store_filename = 'data/test-%s-%s-%s-%s-%s-%s-%s.nc' % (system_name, integrator_name, str(switching_flag), platform_name, precision_model, '%.1e' % constraint_tolerance, '%.1e' % pme_tolerance)
    text_filename = 'data/test-%s-%s-%s-%s-%s-%s-%s.txt' % (system_name, integrator_name, str(switching_flag), platform_name, precision_model, '%.1e' % constraint_tolerance, '%.1e' % pme_tolerance)

    # Skip if we already have written this file.
    if os.path.exists(store_filename) and os.path.exists(text_filename):
        continue

    #=============================================================================================
    # Select platform.
    #=============================================================================================

    platform = openmm.Platform.getPlatformByName(platform_name)
    #deviceid = rank % ngpus
    deviceid = 0
    if platform_name == 'CUDA':
        #platform.setPropertyDefaultValue('CudaDeviceIndex', '%d' % deviceid) # select Cuda device index    
        platform.setPropertyDefaultValue('CudaPrecision', precision_model)        
    elif platform_name == 'OpenCL':
        #platform.setPropertyDefaultValue('OpenCLDeviceIndex', '%d' % deviceid) # select OpenCL device index
        platform.setPropertyDefaultValue('OpenCLPrecision', precision_model)        
    print "node %3d using GPU %d platform %s precision %s" % (rank, deviceid, platform_name, precision_model)

    #=============================================================================================
    # Get positions and velocities from equilibrated NetCDF file. 
    #=============================================================================================

    ncfile = netcdf.Dataset('data/system-%s-%s.nc' % (system_name, switching_flag), 'r')
    positions = units.Quantity(ncfile.variables['positions'][:,:], units.nanometers)
    velocities = units.Quantity(ncfile.variables['velocities'][:,:], units.nanometers / units.picoseconds)
    box_vectors = units.Quantity(ncfile.variables['box_vectors'][:,:], units.nanometers)
    serialized_system = str(ncfile.variables['system'][0])    
    system = openmm.XmlSerializer.deserialize(serialized_system)
    ncfile.close()

    # Set PME tolerance
    set_pme_tolerance(system, pme_tolerance)

    #=============================================================================================
    # Open NetCDF file for writing.
    #=============================================================================================

    print "Node %d: Opening '%s' for writing..." % (rank, store_filename)
    ncfile = netcdf.Dataset(store_filename, 'w', version='NETCDF4')
    initialize_netcdf(ncfile, system, ntimesteps_to_try, nrecords)

    #=============================================================================================
    # Run production simulation, recording total energy RMS.
    #=============================================================================================
 
    outfile = open(text_filename, 'w') 
    output = '%s : integrator %s | switch %s | platform %s | precision %s | constraint tolerance %s | pme tolerance %s' % (system_name, integrator_name, str(switching_flag), platform_name, precision_model, '%.1e' % constraint_tolerance, '%.1e' % pme_tolerance)
    print output
    outfile.write(output + '\n')
    last_rms_total_energy = None
    for timestep_index in range(ntimesteps_to_try):
        timestep = timesteps_to_try[timestep_index]

        # Compute number of steps per record.
        nsteps_per_record = int(record_interval / timestep)

        #print "node %3d starting production simulation for timestep %.3f fs..." % (rank, timestep / units.femtoseconds)

        # Initialize integrator.
        if integrator_name == 'VerletIntegrator':
            integrator = openmm.VerletIntegrator(timestep)
        elif integrator_name == 'VelocityVerletIntegrator':
            from openmmtools.integrators import VelocityVerletIntegrator
            integrator = VelocityVerletIntegrator(timestep)
        else:
            raise Exception("Integrator '%s' unknown." % integrator_name)

        # Set constraint tolerance.
        integrator.setConstraintTolerance(constraint_tolerance)

        # Create Context.
        context = openmm.Context(system, integrator, platform)
        context.setPeriodicBoxVectors(box_vectors[0,:], box_vectors[1,:], box_vectors[2,:])
        context.setPositions(positions)
        context.setVelocities(velocities) 

        # Run integrator to eliminate any initial artifacts.
        integrator.step(nsteps_per_record)

        total_energy_n = numpy.zeros([nrecords], numpy.float64)

        for record in range(nrecords):
            # Run integrator.
            integrator.step(nsteps_per_record)

            # Compute total energy.
            state = context.getState(getEnergy=True)
            kinetic_energy = state.getKineticEnergy()
            potential_energy = state.getPotentialEnergy()
            total_energy = kinetic_energy + potential_energy

            # Store total energy.
            total_energy_n[record] = total_energy / kT
            
        # Store summary statistics.
        rms_total_energy = numpy.std(total_energy_n)
        total_energy_drift = (total_energy_n[-1] - total_energy_n[0]) / (simulation_length/units.nanoseconds)

        ncfile.variables['total_energy'][timestep_index,:] = total_energy_n[:]
        ncfile.variables['rms_total_energy'][timestep_index] = rms_total_energy
        ncfile.sync()

        output = "timestep %8.3f fs | RMS total energy %20.8f kT | drift %20.8f kT" % (timestep / units.femtoseconds, rms_total_energy, total_energy_drift)
        print output
        if last_rms_total_energy:
            factor = rms_total_energy / last_rms_total_energy
            outfile.write('%8.3f %24.8e %24.8e %8.3f' % (timestep/units.femtoseconds, rms_total_energy, total_energy_drift, factor))
        else:
            outfile.write('%8.3f %24.8e %24.8e %8s' % (timestep/units.femtoseconds, rms_total_energy, total_energy_drift, ''))
        last_rms_total_energy = rms_total_energy

        if (total_energy_drift > 1.0):
            outfile.write('      ***')

        outfile.write('\n')
        
        # Clean up
        del context, integrator

    # Close output files.
    outfile.close()
    ncfile.close()



