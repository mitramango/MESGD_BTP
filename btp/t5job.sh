#! /bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=3-00:00:00
#SBATCH --partition=testp
##SBATCH --error=/nlsasfs/home/ttbhashini/arroy/bishal/logs/coral_%x.%J.err
#SBATCH --output=/nlsasfs/home/ttbhashini/arroy/bishal/logs/coral_%x.%J.out
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Job id is $SLURM_JOBID"
echo "Job submission directory is : $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

#################conda environment path ################################
source /nlsasfs/home/ttbhashini/sbishal/anaconda3/envs/mesgd/bin/python
gpustat

# Activate
conda activate mesgd

# Workdir
cd /nlsasfs/home/ttbhashini/sbishal/mangalik/MESGD_BTP/btp

# Internet
export http_proxy=http://proxy-10g.10g.siddhi.param:9090
export https_proxy=http://proxy-10g.10g.siddhi.param:9090
export ftp_proxy= http://proxy-10g.10g.siddhi.param:9090

# wandb
# export WANDB_API_KEY=...
# export WANDB_ENTITY=...


# Remember the diff
# git diff

# Run script

python run.py
