## AortaSeg24 challenge
The code provided particpated in the AortaSeg24 challenge (https://aortaseg24.grand-challenge.org/). The code heavily depends on the nnUNet framework by Isensee (https://github.com/MIC-DKFZ/nnUNet). We added a custom Trainer which does not use mirroring to differentiate between left and right arteries as well as a custom loss combining Dice, TopK and Skeleton Recall loss. Due to the fact that at time of the start of the challenge we were not aware of the publicly available code (https://github.com/MIC-DKFZ/Skeleton-Recall) we did the implementation ourselfs according to the paper (https://arxiv.org/abs/2404.03010).


## Instructions

### 1. Download and Unzip
Download and unzip the provided folder. Then, download and unzip the model weights from the following link:  
[Download Weights](https://fileshare.uibk.ac.at/d/1f307883aec746028f24/)

Ensure the weights are stored in the following structure:

nnUNet_results/Dataset504_aorta/ nnUNetTrainerNoMirroring.../fold_X

Place the entire `nnUNet_results` folder into the `resources` subfolder of the provided directory.

---

### 2. Set Up the Environment

Create a new environment using your preferred environment manager (e.g., Conda or virtualenv), and activate the environment.

Next, install PyTorch and CUDA using the following command:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install modified nnUNet
Now, install the modified version of nnUNet provided in the resources folder:

```bash
pip install -e path_to/resources/nnUNET/
```
### 4. Set Environment Variables
SEt the rquired paths for nnUNet:

```bash
export nnUNet_raw="/path_to/resources/nnUNet_raw"
export nnUNet_results="/path_to/resources/nnUNet_results"
export nnUNet_preprocessed="/path_to/resources/nnUNet_preprocessed"
```

### 5. Prepare the Input Data
Place the volume you want to segment in the following folder:
```bash
test/input/images/ct-angiography
```

### 6. Run the Inference
To run the inference, use the following command:
```bash
python inference_challenge.py
```
 The script processes the first .tiff and .mha file found in the input folder and saves the result as output.mha in the following directory:
```bash
test/output/images/aortic-branches
```
