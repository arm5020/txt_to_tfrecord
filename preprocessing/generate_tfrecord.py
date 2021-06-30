# Sample Tensorflow txt-to-TFRecord converter

# generate_tfrecord.py [-h ] [-x txt_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]

import os
import glob
import pandas as pd
import numpy as np
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] =  '2'
import tensorflow.compat.v1 as tf


# Argument Parser

parser = argparse.ArgumentParser(
        description="Sample Tensorflow txt-to-TFRecord converter")
parser.add_argument("-x", "--txt_dir", help="Path to where the .txt input files are stored.", type=str)
parser.add_argument("-l", "--labels_path", help="Path to the labels (.pbtxt) file.", type=str)
parser.add_argument("-o", "--output_path", help="Path of output TFRecord (.record) file.", type=str)
parser.add_argument("-i", "--image_dir", help="Path to the folder where the input image files are stored. ", type=str, default=None)
parser.add_argument("-c", "--csv_path", help="Path of output .csv file. If none provided, then no file will be written", type=str, default=None)

args = parser.parse_args()

if args.image_dir is None:
        args.image_dir = args.txt_dir


def txt_to_csv(path):
    # Iterates through txt files in a given dir and combines them into a single Pandas DF
    # params: path : str
    # returns: Pandas Dataframe

    txt_list = []
    column_names = ['filename', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'difficult', 'category']
    df = pd.DataFrame(columns = column_names);
    for txt_file in glob.glob(path+ '/*.txt'):
        f = open(txt_file)
        x = 0
        fname = os.path.basename(f.name)
        print(fname)
        for line in f:
            if x == 0:
                parts = line.rstrip('\n').split(':')
                source = parts[1]
                x = x + 1;
            elif x == 1:
                parts2 = line.rstrip('\n').split(':')
                gsd = parts2[1]
                x = x + 1;
            else:
                objdata = line.split(' ')
                x1 = objdata[0]
                y1 = objdata[1]
                x2 = objdata[2]
                y2 = objdata[3]
                x3 = objdata[4]
                y3 = objdata[5]
                x4 = objdata[6]
                y4 = objdata[7]
                category = objdata[8]
                difficult = objdata[9]
                xs = [x1, x2, x3, x4]
                xmin =int(min(xs))
                xmax = int(max(xs))
                ys = [y1, y2, y3, y4]
                ymin = int(min(ys))
                ymax = int(max(ys))
                width = xmax - xmin
                height = ymax - ymin
                df.loc[len(df.index)] = [fname, width, height, xmin, ymin, xmax, ymax, difficult, category]
    return df

def create_tf_example(features, label):

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[features[1].encode('utf-8')])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[features[2]])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[features[3]])),
        'xmin': tf.train.Feature(int64_list=tf.train.Int64List(value=[features[4]])),
        'ymin': tf.train.Feature(int64_list=tf.train.Int64List(value=[features[5]])),
        'xmax': tf.train.Feature(int64_list=tf.train.Int64List(value=[features[6]])),
        'ymax': tf.train.Feature(int64_list=tf.train.Int64List(value=[features[7]])),
        'difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=[features[8]]))
    }))
    return tf_example

def main():
    df = txt_to_csv(args.txt_dir)
    df.to_csv('dataset.csv', encoding='utf-8')
    csv = pd.read_csv('dataset.csv').values
    with tf.python_io.TFRecordWriter(args.output_path) as writer:
        for row in csv:
            features, label = row[:-1], row[-1]
            print (features, label)
            example = create_tf_example(features, label)
            writer.write(example.SerializeToString())
    writer.close()

main()
