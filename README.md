# Automatic Human Activity Recognition System Using Clustering

This repository contains the code and dataset for an automatic human activity recognition system using sensor data. The objective of this study is to design and evaluate a system for automatic recognition of human activity with the minimum number of devices and computational resources available. you can read the full description and results of the project in the `Cluster Human Activities.pdf` file. 

## Dataset

The dataset used in this research consists of sensor data collected from multiple locations on the body, including the torso (T), right arm (RA), left arm (LA), right leg (RL), and left leg (LL), at a frequency of 25 Hz, using 3-axis accelerometers, gyroscopes, and magnetometers. The data collected results in 45 values per sample (sensor features). The dataset is publicly available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities). 

The activities dataset includes 8 participants in total, each of whom performed a set of 19 activities for 5 minutes each. In this research, we are going to specifically focus on the subject number 2 activities. The data is then divided into 5-second segments, resulting in files with 125 rows and 45 columns. 

## Installation

To run this code, the following packages need to be installed: (It is also recommended to open the code in jupyter notebooks to adjust different parameters.

```
pip install -r requirements.txt
```

## Usage
To run the code, execute the following command in the terminal:

```
python main.py
```

## Conclusion

At the end, we have designed and evaluated an automatic human activity recognition system that uses a combination of data pre-processing algorithms and K-Means clustering technique to analyze the sensor data and classify the performed activity with high accuracy using clustering algorithms. The final selection consisted of 5 sensors and 7 features, which was able to predict human activities with a high level of accuracy. Despite some limitations, the results obtained using the K-Means algorithm were promising.
