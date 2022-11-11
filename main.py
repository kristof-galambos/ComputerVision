"""
todo:
    mostly done - implement primitive DNN and CNN
    done - upload to github, create gitignore
    run once every model with many epochs, long processing for final results
"""

from celeba_preprocess import preprocess
from train_model import train_vgg, train_cnn, train_dnn
from classify import classification

PREPROCESS = True
TRAIN_MODEL = True
CLASSIFY = True
MODEL_TYPE = 'vgg'
# MODEL_TYPE = 'cnn'
# MODEL_TYPE = 'dnn'

if PREPROCESS:
    preprocess(play=True)

if TRAIN_MODEL:
    if MODEL_TYPE == 'cnn':
        train_cnn(data_path='/Users/kristofgalambos/Downloads/archive/train_play',
                  model_path='models/mac_model_celeba_cnn_playing', epochs=6)
    elif MODEL_TYPE == 'dnn':
        train_dnn(data_path='/Users/kristofgalambos/Downloads/archive/train_play',
                  model_path='models/mac_model_celeba_dnn_playing', epochs=12)
    else:
        train_vgg(data_path='/Users/kristofgalambos/Downloads/archive/train_play',
                  model_path='models/mac_model_celeba_playing', epochs=1)

if CLASSIFY:
    if MODEL_TYPE == 'cnn':
        classification('models/mac_model_celeba_cnn_playing')
    elif MODEL_TYPE == 'dnn':
        classification('models/mac_model_celeba_dnn_playing')
    else:
        classification('models/mac_model_celeba_playing')

print('PROGRAM ENDS HERE')
