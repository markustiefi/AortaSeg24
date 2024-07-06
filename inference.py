import inspect
import itertools
import multiprocessing
import os
from pathlib import Path
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional
import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


def predict_entry_point(input_folder:str, output_folder:str, folds: list):
    npp = 1
    nps =1
    save_probabilities = False
    prev_stage_predictions = None
    num_parts = 1
    part_id = 0
    continue_prediction = False

    id_ = '504'
    tr = 'nnUNetTrainerNoMirroringDiceTopKSkelRecall'
    plans = 'nnUNetPlans'
    configuration = '3d_fullres'

    tile_step_size = 0.3
    disable_tta = True
    verbose = True
    disable_progress_bar = False

    print(
        "Minor changes thus: \n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    if not isdir(output_folder):
        maybe_mkdir_p(output_folder)

    model_folder = get_output_folder(id_, tr, plans, configuration)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)


    if device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')
    
    predictor = nnUNetPredictor(tile_step_size=tile_step_size,
                                use_gaussian=True,
                                use_mirroring=not disable_tta,
                                perform_everything_on_gpu=True,
                                device=device,
                                verbose=verbose,
                                verbose_preprocessing=verbose,
                                allow_tqdm=not disable_progress_bar)
    
    predictor.initialize_from_trained_model_folder(
        model_folder,
        folds,
        checkpoint_name= 'checkpoint_final.pth'
    )

    predictor.predict_from_files(input_folder, output_folder, save_probabilities=save_probabilities,
                                 overwrite=not continue_prediction,
                                 num_processes_preprocessing=npp,
                                 num_processes_segmentation_export=nps,
                                 folder_with_segs_from_prev_stage=prev_stage_predictions,
                                 num_parts=num_parts,
                                 part_id= part_id)
    

if __name__ == '__main__':
    import sys
    import os
    from nnunetv2.paths import nnUNet_results, nnUNet_raw
    
    input_folder = './input/images/ct-angiography'
    output_folder = './output/images/aortic-branches'
    predict_entry_point(input_folder = input_folder, output_folder = output_folder, folds = [1])
