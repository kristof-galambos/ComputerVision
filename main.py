"""
todo:
    implement primitive DNN and CNN
    done - upload to github, create gitignore
"""

from celeba_preprocess import preprocess
from train_model import training
from classify import classification

PREPROCESS = True
TRAIN_MODEL = True
CLASSIFY = True

if PREPROCESS:
    preprocess(play=True)

if TRAIN_MODEL:
    training(data_path='/Users/kristofgalambos/Downloads/archive/train_play', model_path='models/mac_model_celeba_playing', epochs=1)

if CLASSIFY:
    classification('models/mac_model_celeba_playing')

print('PROGRAM ENDS HERE')
