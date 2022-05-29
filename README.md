# CRNet: Consistent and Relational Feature Learning for Weakly Supervised Temporal Action Localization

## Content

- [Dependencies](#dependencies)
- [Code and Data Preparation](#code-and-data-preparation)
- [Training](#training)
- [Testing](#testing)



## Dependencies

Please make sure Python>=3.6 is installed ([Anaconda3](https://repo.anaconda.com/archive/) is recommended).

Required packges are listed in requirements.txt. You can install them by running:

```
pip install -r requirements.txt
```

## Code and Data Preparation



Prepare the features.

   * Here, we provide the two-stream I3D features for THUMOS'14. You can download them from [Google Drive](https://drive.google.com/file/d/1paAv3FsqHtNsDO6M78mj7J3WqVf_CgSG/view?usp=sharing) or [Weiyun](https://share.weiyun.com/fQRZnfJq).
   * Unzip the downloaded features into the `data` folder. Make sure the data structure is as below.
   
   ```
   ├── data
   └── THUMOS14
       ├── gt.json
       ├── split_train.txt
       ├── split_test.txt
       └── features
           ├── ...
   ```
   
   * Note that these features are originally from [this repo](https://github.com/Pilhyeon/BaSNet-pytorch).



## Training 

You can use the following command to train CRNet:

```
python main.py train
```

After training, you will get the results listed in [this table](#table_result).

## Testing 

You can evaluate a trained model by running:

```
python main.py test MODEL_PATH
```

Here, `MODEL_PATH` denotes for the path of the trained model.






