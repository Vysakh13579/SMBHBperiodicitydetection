#!/bin/bash

#SBATCH --partition=all        # Request nodes from the “all” partition
#SBATCH --nodes 1              # How many nodes to ask for same as SBATCH -N 1 ?
#SBATCH --ntasks  1            # Number of tasks (MPI processes)
#SBATCH --cpus-per-task 1      # Number of logical CPUS (threads) per task
#SBATCH --time 0-3:00:00       # How long you need to run for in days-hours:minutes:seconds
#SBATCH --mem 16gb             # How much memory you need per node
#SBATCH -J GP_JAXNS               # The job name. If not set, slurm uses the name of your script
#SBATCH --output=myjob_%j.out
#SBATCH --error=myjob_%j.err



function clean_up {
    echo "### Running Clean_up ###"
    # copy the data off of /hddstore and onto filer0 (home directory)
    cp -rf $OUTPUT_FOLDER/results/json_files/highALPHA_2_4	/home/$USER/results/json_files/
    # remove the folder from /hddstore
    rm -rf $OUTPUT_FOLDER
    echo "Removed all files"
    echo "Finished"; date
    exit
}

# call "clean_up" function when this script exits, it is run even if SLURM cancels the job
trap 'clean_up' EXIT

module purge
module load anaconda3/2021-05

conda activate 14444429env

mkdir -p /hddstore/$USER
export OUTPUT_FOLDER=$(mktemp -d -p /hddstore/$USER)
echo $OUTPUT_FOLDER $SLURMD_NODENAME

mkdir $OUTPUT_FOLDER/results/
mkdir -p $OUTPUT_FOLDER/simDATAcsvs/
cp /home/$USER/NSmodels.py $OUTPUT_FOLDER/NSmodels.py
cp /home/$USER/batch_run.py $OUTPUT_FOLDER/batch_run.py
cp /home/$USER/THESIS.py $OUTPUT_FOLDER/THESIS.py
cp /home/$USER/simDATAcsvs/simDATA_highALPHA_2_4.csv $OUTPUT_FOLDER/simDATAcsvs/simDATA_highALPHA_2_4.csv

mkdir -p $OUTPUT_FOLDER/results/json_files/highALPHA_2_4/

cd $OUTPUT_FOLDER

python batch_run.py

