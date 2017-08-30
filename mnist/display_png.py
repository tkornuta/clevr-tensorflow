# Simple script loading the png images using TF.
# Displays image using Pillow (PIL).
# Based on the example from:
# https://stackoverflow.com/questions/33648322/tensorflow-image-reading-display  
# author: tkornuta

import sys
import tensorflow as tf
from PIL import Image
import numpy as np

# Filename
image = '../../data/clevr/training_data/train_subset_000/images/CLEVR_train_000000.png'

#  list of files to read
filename_producer = tf.train.string_input_producer([image])

# Create a file reader:
# https://www.tensorflow.org/api_docs/python/tf/WholeFileReader
reader = tf.WholeFileReader()
# Read the file.
key, value = reader.read(filename_producer)

# Use the TF png decoder and "load" the image from the file.
my_img = tf.image.decode_png(value) 

# Initialize a session.
# https://www.tensorflow.org/api_guides/python/image#Encoding_and_Decoding
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)

  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  # Show image and its shape (dimensions).
  image = my_img.eval() 
  print(image.shape)
  Image.fromarray(np.asarray(image)).show()

  coord.request_stop()
  coord.join(threads)
