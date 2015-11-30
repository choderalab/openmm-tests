#!/bin/tcsh
#  Batch script for mpirun job on cbio cluster.
#
#
# walltime : maximum wall clock time (hh:mm:ss)
#PBS -l walltime=72:00:00
#
# join stdout and stderr
#PBS -j oe
#
# spool output immediately
#PBS -k oe
#
# specify queue
#PBS -q gpu
#
# nodes: number of 8-core nodes
#   ppn: how many cores per node to use (1 through 8)
#       (you are always charged for the entire node)
##PBS -l nodes=32,tpn=1,gpus=1:shared:gtxtitan
#PBS -l nodes=3:ppn=4:gpus=4:exclusive
#
# export all my environment variables to the job
##PBS -V
#
# job name (default = name of script file)
#PBS -N rms
#
# specify email
#PBS -M jchodera@gmail.com
#
# mail settings
#PBS -m n
#
# filename for standard output (default = <job_name>.o<job_id>)
# at end of job, it is in directory from which qsub was executed
# remove extra ## from the line below if you want to name your own file
#PBS -o /cbio/jclab/home/chodera/vvvr/openmm/timescale-correction/vvvr.out

cd $PBS_O_WORKDIR

echo | grep PYTHONPATH
which python

cat $PBS_GPUFILE

# Only use one OpenMM CPU thread.
setenv OPENMM_CPU_THREADS 1

date
build_mpirun_configfile  --mpitype=conda "python test_energy_rms.py"
mpirun -configfile configfile
date

