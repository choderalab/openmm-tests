#!/bin/tcsh
#  Batch script for mpirun job on cbio cluster.
#
#
# walltime : maximum wall clock time (hh:mm:ss)
#PBS -l walltime=24:00:00
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
##PBS -l nodes=1:ppn=1:gpus=1:shared:gtxtitan
##PBS -l nodes=1:ppn=1:gpus=1:shared:gtx680
#PBS -l nodes=10,tpn=1,gpus=1:shared:gtx680
##PBS -l nodes=1:ppn=4:gpus=4:shared:gpu-1-12
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

setenv VVVR_DIR $PBS_O_WORKDIR

echo | grep PYTHONPATH

#canopy
#openmm-git

which python

# Only use one OpenMM CPU thread.
setenv OPENMM_CPU_THREADS 1

date
mpirun -rmk pbs python test_energy_rms.py
date

