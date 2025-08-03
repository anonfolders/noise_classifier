# Noise classifier CNN

This repo contains the instructions to setup, train, and test a noise classifier.

## Setup

### Environment

Install the necessary libraries using command <code>pip install -r requirements.txt</code>

### Data

This step shows how to organize the data to be loaded for use. In this work, we use the library data structure in Python to store the location of the of the PCD and store in data_cfg.py.
We will go over an example to setup KITTI sequence 09. After the previous RQ, we would have several folders containing the clean (original) data and various augmented data. Thus, for each class of noise that we are interested in, we will create a dictionary as follows, where the key is the severity and the value is the path to the folder containing the corresponding PCD. For example, the dictionary for clean and fog data for sequence 09 is as follows:

```
KITTI_ODO_09_CLEAN = {
    '01': 'data/dataset/sequences/09/velodyne',
    '02': 'data/dataset/sequences/09/velodyne',
    '03': 'data/dataset/sequences/09/velodyne',
    '04': 'data/dataset/sequences/09/velodyne',
    '05': 'data/dataset/sequences/09/velodyne',
}

KITTI_ODO_09_FOG = {
    '01': 'data/kitti_fog_1/dataset/sequences/09/velodyne',
    '02': 'data/kitti_fog_2/dataset/sequences/09/velodyne',
    '03': 'data/kitti_fog_3/dataset/sequences/09/velodyne',
    '04': 'data/kitti_fog_4/dataset/sequences/09/velodyne',
    '05': 'data/kitti_fog_5/dataset/sequences/09/velodyne',
}
```
In the paper, we mentioned merging rain and snow which can be done as follows.
```
KITTI_ODO_09_RAIN_SNOW = {
    '01': 'data/kitti_rain_1/dataset/sequences/09/velodyne',
    '02': 'data/kitti_rain_2/dataset/sequences/09/velodyne',
    '03': 'data/kitti_rain_3/dataset/sequences/09/velodyne',
    '04': 'data/kitti_rain_4/dataset/sequences/09/velodyne',
    '05': 'data/kitti_rain_5/dataset/sequences/09/velodyne',
    '06': 'data/kitti_snow_1/dataset/sequences/09/velodyne',
    '07': 'data/kitti_snow_2/dataset/sequences/09/velodyne',
    '08': 'data/kitti_snow_3/dataset/sequences/09/velodyne',
    '09': 'data/kitti_snow_4/dataset/sequences/09/velodyne',
    '10': 'data/kitti_snow_5/dataset/sequences/09/velodyne',
}
```
We repeat this process for all classes needed for classification. Examples for sequences 09 and 10 are included in data_cfg.py. After locating all PCD folders, we create an overall dictionary pointing to all of them for each sequence as follows, where the key are the class name and the value is the corresponding dictionary.
```
KITTI_ODO_09 = {
    'clean': KITTI_ODO_09_CLEAN,
    'fog': KITTI_ODO_09_FOG,
    'rain_snow': KITTI_ODO_09_RAIN_SNOW,
    'noise_uniform_gaussian': KITTI_ODO_09_NOISE_UNIFORM_GAUSSIAN,
    'noise_background': KITTI_ODO_09_NOISE_BACKGROUND,
    'density_dec': KITTI_ODO_09_DENSITY_DEC,
    'density_inc': KITTI_ODO_09_DENSITY_INC,
}
```

## Training, validation, and testing
After setup we can simply run the follwing command to train/test for the specific sequences.

```python main.py <function> <dataset>```

where `<function>` is either `train` or `test` and `<dataset>` can be either `08`, `09`, or `10`.

Following the example in the paper, we can train the CNN on sequence 09 as follows.

```python main.py train 09```

which will generate a model `noise_classifier_weights.pth` while showing the accuracy on the validation split. We can then test on sequence 10 as follows.

```python main.py test 10```

which will show the testing result including the accuracy for each class.