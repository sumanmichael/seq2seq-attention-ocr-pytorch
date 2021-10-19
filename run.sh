#!/bin/bash
#SBATCH --time=6-00:00:00
#SBATCH -A sumanmichael
#SBATCH --partition=long
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH -c 20
#SBATCH --mail-user=sumanmichael01@gmail.com
#SBATCH --mail-type=ALL

echo "Running on $(hostname)"

#echo "Loading Cuda Modules" &&
module load cuda/10.2 &&
module load cudnn/7.6.5-cuda-10.2 &&
#echo "Loaded cuda modules" &&

echo "Starting Data Transfer"
echo "-----------------------------------"
echo "Annotations Transfer:"
rsync -az --info=progress2 sumanmichael@ada.iiit.ac.in:/share3/sumanmichael/annotation_txt_files /ssd_scratch/cvit/sumanmichael/ &&
echo "Images Transfer:"
rsync -az --info=progress2 sumanmichael@ada.iiit.ac.in:/share3/bvk_cvit/img_data /ssd_scratch/cvit/sumanmichael/ &&
echo "Codebase Transfer:"
rm -rf /ssd_scratch/cvit/sumanmichael/seq2seq-attention-ocr-pytorch
rsync -az --info=progress2 sumanmichael@ada.iiit.ac.in:/home2/sumanmichael/seq2seq-attention-ocr-pytorch /ssd_scratch/cvit/sumanmichael/ &&
echo "Data Transfer successful"

echo "-----------------------------------"

source /home2/sumanmichael/miniconda3/etc/profile.d/conda.sh &&
conda activate pl &&
echo "conda env Activated!" &&

cd /ssd_scratch/cvit/sumanmichael/seq2seq-attention-ocr-pytorch &&
echo "-------------------------------"
echo " CUDA VERSIONS "
echo "--------nvidia-smi-------------"
nvidia-smi
echo "--------nvcc-------------------"
nvcc --version
echo "---------pytorch-cuda----------"
python -c "import torch; print('TORCH_CUDA_V:',torch.version.cuda)"
echo "-------------------------------"

CUDA_LAUNCH_BLOCKING=1;HYDRA_FULL_ERROR=1 python train.py logger.name="$1"
