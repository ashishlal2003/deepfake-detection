# Deepfake Detection

## Introduction
This project is a deepfake detection project. The goal is to detect deepfake image using deep learning. The dataset used in this project is from [Kaggle](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection). The dataset contains 1081 real images and 960 fake images. The dataset is divided into training and testing set in a ratio of 80:20. The model used in this project is a Convolutional Neural Network (CNN) model. The model is trained using the training set and tested using the testing set.

## Dataset
Store the downloaded images in a new folder in the root directory of the project called `data`. The folder should have the following structure:
```
data
│
└─── fake
└─── real
```

## Requirements
Create a virtual environment and install the required packages using the following commands:

```bash
python -m venv venv
source venv/bin/activate # For Linux
venv\Scripts\activate # For Windows
pip install -r requirements.txt
```

## Model
You may use the pre-trained model from [Google Drive](https://drive.google.com/file/d/1SlyGpj3PnPdEHd9nd_5WrI3Iwxi0I3kn/view?usp=sharing) or train the model yourself using the below steps.

Run the following command to make the dataset ready for model training:

```bash
python scripts/preprocess.py
python scripts/split.py
python scripts/reorganize.py
```

Run the following command to train the model:

```bash
python scripts/train.py
```

The trained model will be saved in the `models` folder.

## Inferencing
Run the following command to test the model (example added):

```bash
python scripts/infer.py --model models/model.pth --image data/real/real_00001.jpg
```

## Conclusion
The model can be further improved by using more data and more complex models. It is under development and will be updated in the future.

## Author
- [Ashish Lal](https://www.github.com/ashishlal2003)