## Instructions

### 1. Download and Unzip
Download and unzip the provided folder. Then, download and unzip the model weights from the following link:  
[Download Weights](https://fileshare.uibk.ac.at/d/1f307883aec746028f24/)

Ensure the weights are stored in the following structure:

Place the entire `nnUNet_results` folder into the `resources` subfolder of the provided directory.

---

### 2. Set Up the Environment

Create a new environment using your preferred environment manager (e.g., Conda or virtualenv), and activate the environment.

Next, install PyTorch and CUDA using the following command:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e path_to/resources/nnUNET/

export nnUNet_raw="/path_to/resources/nnUNet_raw"
export nnUNet_results="/path_to/resources/nnUNet_results"
export nnUNet_preprocessed="/path_to/resources/nnUNet_preprocessed"

test/input/images/ct-angiography

python inference_challenge.py

test/output/images/aortic-branches
