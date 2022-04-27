# ECE 740 LEC 802 Course Project: Object Tracking in Video with SiamFC model based on MindSpore

This repository contains the code for the course project of ECE 740 LEC 802 Advanced Topics in Signal and Image Processing: Deep Learning in Computer Vision. In the project we trained and tested the SiamFC tracker using GOT-10k and VOT2018 datasets, respectively.

Team members: Yafei Ou, Kirill Makhachev

## Requirements

GOT-10k dataset is needed for training. The dataset should be placed in the folder "dataset". VOT2018 dataset is needed for testing and should be placed in the folder "VOT2018" (Or it can be automatically downloaded when running `test.py`). 

Python 3.7 is needed. **MindSpore version 1.2.0** (**the latest is 1.6.1**) is used along with the Huawei Ascend NPU provided by the Huawei Atlas AI platform (atlas.cmc.ca). Some other requirements include:

```
numpy==1.17.5
opencv-python==4.4.0.46
got10k==0.1.3
```

Unfortunately, we could not test the code locally using a CPU or GPU. Some changes to the code might be needed before running it locally.

```bash
.
├── dataset\        # training dataset GOT-10k
│   ├── test\
│   ├── train\
│   ├── val\
├── VOT2018\        # test set VOT2018
│   ├── ants1\
│   ├── ants3\
│   ├── ...
│   ├── ...
├── models\
│   ├── siamfc\     # for storing trained models
├── pysot-toolkit   # pysot-toolkit source code
├── results
│   ├── VOT2018\    # evaluation results for VOT2018
├── alexnet.py
├── config.py
├── custom_transforms.py
├── dataset.py
├── README.md
├── report_VOT18.sh
├── ...
├── ...
```



## Train

Use the following command to start training in background and log the output to `log_train.txt`.

```bash
bash run_train.sh
```

## Evaluation

Use the following code.

```bash
bash run_eval.sh
```

This will generate the predicted results for the VOT2018 dataset in the folder "results". Then use the pysot-toolkit for calculating the performance scores. Make sure that the file `VOT2018.json`  is in the folder "VOT2018".

```bash
nano report_VOT18.sh # use nano editor
```

Change the `--dataset_dir` and `--tracker_result_dir` arguments to their corresponding locations (the default ones are for our specific case). Then run the script to generate the scores.

```bash
bash report_VOT18.sh

# you will see
loading VOT2018: 100%|██████████████████████████████████| 60/60 [00:02<00:00, 27.64it/s, zebrafish1]
eval ar: 100%|████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.43s/it]
eval eao: 100%|███████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.96s/it]
------------------------------------------------------------
|Tracker Name| Accuracy | Robustness | Lost Number |  EAO  |
------------------------------------------------------------
|   SiamFC   |  0.511   |   0.534    |    114.0    | 0.222 |
------------------------------------------------------------
```

