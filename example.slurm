#!/bin/bash
#SBATCH --job-name=example
#SBATCH --chdir=/cluster/datastore/your_user/your_project
#SBATCH --account=master        
#SBATCH --mem=128GB             
#SBATCH --nodelist=hpc1
#SBATCH --ntasks=1              # one python process manages paralellism 
#SBATCH --cpus-per-task=64      # many processes are spawned using a certain number of cores per task
#SBATCH --gres=gpu:2
#SBATCH --output=out/logging/last_run.log

run_stuff.sh > my.log 2>&1
