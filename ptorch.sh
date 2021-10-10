#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH -A bvk_cvit
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -c 20
#SBATCH --mail-user=khadiravana.belagavi@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --nodelist gnode11
echo "Running on $(hostname)"

#echo "Loading Cuda Modules" &&
module load cuda/10.2 &&
module load cudnn/7.6.5-cuda-10.2 &&
#echo "Loaded cuda modules" &&

echo "Starting Data Transfer"
echo "-----------------------------------"
echo "Annotations Transfer:"
rsync -az --info=progress2 bvk_cvit@ada.iiit.ac.in:/share3/bvk_cvit/annotation_txt_files /ssd_scratch/cvit/bvk_cvit/ &&
echo "Images Transfer:"
rsync -az --info=progress2 bvk_cvit@ada.iiit.ac.in:/share3/bvk_cvit/img_data /ssd_scratch/cvit/bvk_cvit/ &&
echo "Codebase Transfer:"
rm -rf /ssd_scratch/cvit/bvk_cvit/seq2seq-attention-ocr-pytorch
rsync -az --info=progress2 bvk_cvit@ada.iiit.ac.in:~/seq2seq-attention-ocr-pytorch /ssd_scratch/cvit/bvk_cvit/ &&
echo "Data Transfer successful"

echo "-----------------------------------"

source /home/bvk_cvit/anaconda3/etc/profile.d/conda.sh &&
conda activate pytorch_ocr &&
echo "conda env Activated!" &&

cd /ssd_scratch/cvit/bvk_cvit/seq2seq-attention-ocr-pytorch &&
echo "-------------------------------"
echo " CUDA VERSIONS "
echo "--------nvidia-smi-------------"
nvidia-smi
echo "--------nvcc-------------------"
nvcc --version
echo "---------pytorch-cuda----------"
python -c "import torch; print('TORCH_CUDA_V:',torch.version.cuda)"
echo "-------------------------------"
rm -rf ../train &&
rm -rf ../val &&
python create_dataset.py ../annotation_txt_files/c1_train.txt ../train &&
python create_dataset.py ../annotation_txt_files/val.txt ../val &&
# CUDA_LAUNCH_BLOCKING=1;HYDRA_FULL_ERROR=1 python train.py logger.name="$1"
CUDA_LAUNCH_BLOCKING=1 python train_pt.py --train_list ../train --eval_list ../val 
