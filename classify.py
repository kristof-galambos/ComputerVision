"""
todo:
    done - fix y_proba, change output to be 1 neuron instead of 2
    beautify evaluation of classification: compare 0s and 1s instead of strings like 'male' and 'female'
    implement post-regression activation functions
    fix classification for dnn
    WARNING This does not produce image output. For that, see acomputervision/larger_face.py
"""

import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score


def classification(model_path):
    model = tf.keras.models.load_model(model_path)

    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    # generate data for test set of images
    test_generator = data_generator.flow_from_directory(
        '/Users/kristofgalambos/Downloads/archive/test_play',
        target_size=(178, 218),
        batch_size=1,
        class_mode='binary',
        shuffle=False)
    # obtain predicted activation values for the last dense layer
    test_generator.reset()
    y_proba = np.array(model.predict_generator(test_generator, verbose=1, steps=1000)).flatten()
    # determine the maximum activation value for each sample
    # y_proba = np.array([x[0] for x in pred])  # 1 for male and 0 for female
    y_pred = np.array([round(x) for x in y_proba])
    predicted_class_indices = y_pred
    # label each predicted value to correct gender
    labels = (test_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    # format file names to simply male or female
    filenames = test_generator.filenames
    filenz = [0]
    for i in range(0, len(filenames)):
        filenz.append(filenames[i].split('/')[0])
    filenz = filenz[1:]

    shift = 0 # use different labellings for vgg
    if 'cnn' not in model_path and 'dnn' not in model_path:
        shift = 1
    y_true = np.array([1-shift if x == 'female' else 0+shift for x in filenz])
    print('accuracy_score:', accuracy_score(y_true, y_pred))
    print('auc score:', roc_auc_score(y_true, y_proba))
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    if 'cnn' in model_path:
        fname = 'roc_cnn.pkl'
    elif 'dnn' in model_path:
        fname = 'roc_dnn.pkl'
    else:
        fname = 'roc_vgg.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(fpr, f)
        pickle.dump(tpr, f)
        pickle.dump(thresholds, f)
    plt.figure(1)
    plt.title('ROC curve')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.plot(fpr, tpr)
    xref = np.linspace(0, 1, 10)
    yref = xref
    plt.plot(xref, yref, 'k--')
    plt.savefig('ROC_curve.png')

    # determine the test set accuracy
    match = []
    for i in range(0, len(filenames)):
        match.append(filenz[i] == predictions[i])
    print('accuracy:', match.count(True) / len(filenames))
    results = pd.DataFrame({"Filename": filenz, "Predictions": predictions})
    results.to_csv("GenderID_test_results.csv", index=False)
