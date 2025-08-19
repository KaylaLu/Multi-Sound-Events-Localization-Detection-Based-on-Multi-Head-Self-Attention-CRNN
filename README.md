## Basic Information

This is a Multi-Head Self-Attention CRNN designed for my final project, aligned with master dissertation, aiming to compete for DCASE2024 Task 3 SELD Problem. The code Repository is uploaded on Github.

* Author: Gening Lu
* 
* Affiliation: University of Edinburgh, Acoustics Audio Group
* 
* Last Modified: Aug. 19th, 2025
* 
* Dataset in use: STARSS23, url{https://zenodo.org/records/7880637}
*
* Github address:url{}


### Repository Brief

This repository consists of multiple Python scripts forming one big architecture used to train the MHSA-CRNN (SELDnet) backbone.
* The `batch_feature_extraction.py`: standalone script extracts the features, labels, and normalizes the training and test split features for a given dataset. Make sure you update the location of the downloaded datasets before.
* The `parameter.py`: script consists of all the training, model, and feature parameters.
* The `cls_feature_class.py`: script has routines for labels creation, features extraction and normalization.
* The `cls_data_generator.py`: script provides feature + label data in generator mode for training.
* The `seldnet_model.py`: script implements the MHSA-CRNN architecture.
* The `SELD_evaluation_metrics.py`: script implements the metrics for joint evaluation of detection and localization, derived from DCASE challenge criteria.
* The `train_seldnet.py`: script trains the SELDnet with early-stopping strategy.
* The `cls_compute_seld_results.py` script computes the metrics results on DCASE output format files. 



### Environment Setup

* See SELDnetMac.txt or SELDnetWin.txt; divert these files to .yaml files then import as new environment by Anaconda, depending on the system using for running the project.

* Least environment requirements: Python 3.8, Pytorch 2.2.1.

* Particularly, when using win system, using commandline below:

* Pycharm Terminal activation (substitute step 2 with local address where the conda python is installed):
```
#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
```
#& "D:\SoftwareDownload\Anaconda\Download\shell\condabin\conda-hook.ps1"
```

* To Avoid OMP warning:
```
#$env:KMP_DUPLICATE_LIB_OK="TRUE"
```
```
#python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

* Save the env infos:
```
# conda list --explicit > "E:\Desktop\SELDnet_env.txt"
```



### Training the SELDnet

* Using following commandline in terminal with appropriate conda python environment to extract the features.
```
python3 batch_feature_extraction.py <task-id>
```
or for win system:
```
python batch_feature_extraction.py <task-id>
```


* Then, train the SELDnet with the default parameters using `train_seldnet.py`. 
* 
* Additionally, you can add/change parameters by using a unique identifier \<task-id\>
* 
* in the if-else conditions as seen in the `parameter.py` script and call them as follows.
* 
* Where \<job-id\> is a unique identifier that is used for output filenames (models, training plots).
* 
* You can use any number or string for this.

```
python3 train_seldnet.py <task-id> <job-id>
```
* For Diary Saving with time, run
```
python3 train_seldnet.py <task-id> <job-id> > logs/train_$(date +%Y%m%d_%H%M%S).txt 2>&1
```
* For save the result in another file, run
```
python3 train_seldnet.py <task-id> <job-id> output_log.txt
```




