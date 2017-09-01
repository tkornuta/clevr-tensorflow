import collections
import os
import sys

#from os import path
import random
#import tempfile
#import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import random_seed

from tensorflow.contrib.learn.python.learn.datasets import base
# Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

## Helper functions

def _parse_document(filename):
    """Parses document using space as delimiter."""
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    """Builds and returns a vocabulary for a given document."""
    # Parse document.
    data = _parse_document(filename)
    # Transform data to dictionary (key - value)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    # Returns dictionary that can be used for decoding of the document.
    return word_to_id

def encode_doc_to_one_hot(dense_data_vector, num_classes):
    """Convert data from dense vector of scalars to vector of one-hot vectors."""
    num_labels = len(dense_data_vector)
    result = np.zeros(shape=(num_labels, num_classes))
    result[np.arange(num_labels), dense_data_vector] = 1
    return result.astype(int)

def _extract_document(filename, word_to_id_dict, one_hot=False):
    """Reades a document and encodeds it using a dictionary."""
    data = _parse_document(filename)
    encoded_doc = [word_to_id_dict[word] for word in data if word in word_to_id_dict]
    if one_hot == True:
        return encode_doc_to_one_hot(encoded_doc, len(word_to_id_dict))
    # else: 
    return encoded_doc

## Dataset helper class for storing parsed text.

class TextDataSet(object):

  def __init__(self,
               text,
               phrase_length=100,
               seed=None):
    """Construct a DataSet. Divides (already parsed and encoded) text data into phrases.
    Seed arg provides for convenient deterministic testing.
    """
    # Set seed.
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)

    self._text = text
    self._phrase_length = phrase_length
    self._epochs_completed = 0
    self._index_in_epoch = 0
    
    # Divide document into phrases of a given size.
    doc_size = len(text)
    self._num_examples = int(doc_size/phrase_length)
    # DATA: Process text into phrases.
    self._data = np.array([text[i*phrase_length:(i+1)*phrase_length] for i in range(self._num_examples)])
    # LABELS: Process text into phrases - label is next char, so shifted by one.
    self._labels = np.array([text[i*phrase_length+1:(i+1)*phrase_length+1] for i in range(self._num_examples)])
        
  @property
  def data(self):
    return self._data

  @property
  def labels(self):
    return self._labels

  @property
  def batch_length(self):
    return self._batch_length

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._data = self.data[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      data_rest_part = self._data[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._data = self.data[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      data_new_part = self._data[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((data_rest_part, data_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._data[start:end], self._labels[start:end]

## Function for importing Penn Tree Bank.
def read_ptb(ptb_dir, 
        train,
        valid,
        test,
        phrase_length=100,
        one_hot=False,
        seed=None):

    train_file = os.path.join(ptb_dir, train)
    valid_file = os.path.join(ptb_dir, valid)
    test_file = os.path.join(ptb_dir, test)

    # Build dictionary on the basis of train data.
    word_to_id_vocab = _build_vocab(train_file)
    print (word_to_id_vocab)   

    # Load data.
    train_data = _extract_document(train_file, word_to_id_vocab, one_hot)
    validaton_data = _extract_document(valid_file, word_to_id_vocab, one_hot)
    test_data = _extract_document(test_file, word_to_id_vocab, one_hot)

    options = dict(phrase_length=100,seed=seed)

    # Create datasets.
    train = TextDataSet(train_data, **options)
    validation = TextDataSet(validaton_data, **options)
    test = TextDataSet(test_data, **options)

    return base.Datasets(train=train, validation=validation, test=test)
