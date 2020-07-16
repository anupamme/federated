import numpy as np
import os

from utils import csv_reader as csv
from utils import img_util as img
import densenet

'''
returns (trainX, trainY), (testX, testY)
trainX.shape: 50000, 320, 320, 3
trainY.shape: 50000, 14

testX.shape: 10000, 320, 320, 3
testY.shape: 10000, 14

may have to change the resolution as images may be in some other resolution.

image_folder:
    train.csv
    train/
    valid.csv
    valid/

read train.csv:
    ignore 1st line
    global_nd_array = []
    for each line:
        1st part is image path
        last part is label
        image_nd_array = convert_image(image_path)
        global_nd_array.append()
'''

'''
replace:
    empty by 0
    -1 by 0
'''
def replace_label(item):
    if item == '':
        return 0.0
    else:
        _fitem = float(item)
        if _fitem == -1.0:
            return 1   # multi_class classification
        else:
            return _fitem
        
def filter_labels(labels: list, allowed_indices=[2,6,8,9,10,12]):
    _val = []
    for idx, value in enumerate(labels):
        if idx in allowed_indices:
            _val.append(value)
    return _val

def process_line(parts):
    base_path = '/Volumes/work/data/medical'
#    base_path = '/home/mediratta/'
    rel_path = parts[0]
    label_vec = list(map(lambda x: replace_label(x), filter_labels(parts[5:])))
    image = img.convert_image(os.path.join(base_path, rel_path))
    return image, label_vec

def load_data_sub(_file):
    x_data = []
    x_label = []
    csv_data = csv.read_csv(_file)
    csv_to_use = csv_data[1:]
    for idx, parts in enumerate(csv_to_use):
        if idx == 1000:
            break
        image, label_vec = process_line(parts)
        x_data.append(image)
        x_label.append(label_vec)
    return x_data, x_label

def load_data(image_folder='/Volumes/work/data/medical/CheXpert-v1.0-small'):
    train_file = os.path.join(image_folder, 'train.csv')
    x_train, label_train = load_data_sub(train_file)
    valid_file = os.path.join(image_folder, 'valid.csv')
    x_valid, label_valid = load_data_sub(valid_file)
    return process_data(x_train, label_train), process_data(x_valid, label_valid)

def load_data_gen(image_folder, batch_size):
    train_file = os.path.join(image_folder, 'train.csv')
    gen_train = generate_batch_size(train_file, batch_size)
    valid_file = os.path.join(image_folder, 'valid.csv')
    gen_test = generate_batch_size(valid_file, batch_size)
    return gen_train, gen_test


def process_data(_features, _labels):
    _type = 'float32'
    _features = densenet.preprocess_input(np.array(_features).astype(_type))
    _labels = np.array(_labels).astype(_type)
    return _features, _labels

def generate_batch_size(path:str, batch_size: int):
    csv_data = csv.read_csv(path)
    csv_to_use = csv_data[1:]
    features = []
    target = []
    while True:
        for idx, parts in enumerate(csv_to_use):
            image, label_vec = process_line(parts)
            features.append(image)
            target.append(label_vec)
            if (idx + 1) % batch_size == 0:
                yield process_data(features, target)
                features = []
                target = []
        yield process_data(features, target)
        features = []
        target = []