#!/bin/bash                 
#The previous line is mandatory
 
#SBATCH --job-name=myrun     #Name of your job
#SBATCH --partition=infai_1
#SBATCH --cpus-per-task=8    #Number of cores to reserve
#SBATCH --mem-per-cpu=4G     #Amount of RAM/core to reserve
#SBATCH --time=26:00:00      #Maximum allocated time
#SBATCH --output=myrun.out%j   #Path and name to the file for the STDOUT
#SBATCH --error=myrun.err%j    #Path and name to the file for the STDERR
                 #Load required modules
#%Module torch
source ~/pyVirtualEnv/bin/activate
python DQN_training.py    #Execute your command(s)