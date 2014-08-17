# Test CustomIntegrator precision

from simtk import openmm
from simtk import unit

def PrecisionTestIntegrator(dx):
    integrator = openmm.CustomIntegrator(1.0)
    integrator.addPerDofVariable('dx', dx)
    integrator.addComputePerDof('x', 'x + dx')
    return integrator

# Create System.
system = openmm.System()
system.addParticle(1.0)
positions = [[1.0, 1.0, 1.0]] * unit.nanometers

platform_names = ['Reference', 'CPU', 'OpenCL', 'CUDA']
precision_models = ['single', 'mixed', 'double']

min_decimal_places = 0
max_decimal_places = 20

for platform_name in platform_names:
    for precision_model in precision_models:

        platform = openmm.Platform.getPlatformByName(platform_name)
        if platform_name == 'OpenCL':
            platform.setPropertyDefaultValue('OpenCLPrecision', precision_model)
        elif platform_name == 'CUDA':
            platform.setPropertyDefaultValue('CudaPrecision', precision_model)
        else:
            if precision_model != 'double':
                continue

        # Create Context.
        for decimal_places in range(min_decimal_places, max_decimal_places+1):
            dx = 10**(-decimal_places)
            integrator = PrecisionTestIntegrator(dx)
            context = openmm.Context(system, integrator, platform)
            context.setPositions(positions)
    
            # Take a step.
            integrator.step(1)
    
            # Check change in position.
            state = context.getState(getPositions=True)
            new_positions = state.getPositions()
            dx = (new_positions[0][0] - positions[0][0]) / unit.nanometers

            del context, integrator

            if abs(dx) > 0.0:
                #print "%5d %30.20f" % (decimal_places, dx)
                pass
            else:
                break

        decimal_places -= 1
        print "%12s : %12s : There are %d digits of precision." % (platform_name, precision_model, decimal_places)
        
