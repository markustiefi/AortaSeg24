## AortaSeg24 challenge
The provided code was developed for the [AortaSeg24 Challenge](https://aortaseg24.grand-challenge.org/). It builds upon the nnUNet framework by Isensee ([nnUNet GitHub](https://github.com/MIC-DKFZ/nnUNet)) with custom modifications. Specifically, we implemented a custom trainer that avoids using mirroring to accurately differentiate between the left and right arteries. Additionally, we introduced a custom loss function that combines Dice, TopK, and Skeleton Recall losses.

At the time the challenge started, we were unaware of the publicly available implementation of Skeleton Recall ([Skeleton Recall GitHub](https://github.com/MIC-DKFZ/Skeleton-Recall)), so we independently implemented it based on the corresponding paper ([Skeleton Recall Paper](https://arxiv.org/abs/2404.03010)).

## Usage Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/markustiefi/MedUIBK-AortaSeg24
```

### 2. Download and Unzip
Download and unzip the provided folder. Then, download and unzip the model weights from the following link:  
[Download Weights](https://fileshare.uibk.ac.at/d/1f307883aec746028f24/)

Ensure the weights are stored in the following structure:

nnUNet_results/Dataset504_aorta/ nnUNetTrainerNoMirroring.../fold_X

Place the entire `nnUNet_results` folder into the `resources` subfolder of the provided directory.

---

### 3. Set Up the Environment

Create a new environment using your preferred environment manager (e.g., Conda or virtualenv), and activate the environment.

Next, install PyTorch and CUDA using the following command:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install modified nnUNet
Now, install the modified version of nnUNet provided in the resources folder:

```bash
pip install -e path_to/resources/nnUNET/
```
### 5. Set Environment Variables
Set the required paths for nnUNet (Linux):

```bash
export nnUNet_raw="/path_to/resources/nnUNet_raw"
export nnUNet_results="/path_to/resources/nnUNet_results"
export nnUNet_preprocessed="/path_to/resources/nnUNet_preprocessed"
```

### 6. Prepare the Input Data
Place the volume you want to segment in the following folder:
```bash
test/input/images/ct-angiography
```

### 7. Run the Inference
To run the inference, use the following command:
```bash
python inference_challenge.py
```
 The script processes the first .tiff and .mha file found in the input folder and saves the result as output.mha in the following directory:
```bash
test/output/images/aortic-branches
```
