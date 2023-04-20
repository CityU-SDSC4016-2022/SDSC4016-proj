# Run the main code
source ~/miniconda3/etc/profile.d/conda.sh
source ~/miniconda3/etc/profile.d/mamba.sh
mamba activate tf
export TF_CPP_MIN_LOG_LEVEL=1
python src/fetcher.py