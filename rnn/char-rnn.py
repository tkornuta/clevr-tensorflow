"""
Imports
"""
import numpy as np
import tensorflow as tf
#%matplotlib inline
import matplotlib.pyplot as plt
import time
import os
import urllib.request

FILE_NAME = "/home/tkornuta/data/tiny-shakespeare/tiny-shakespeare.txt"

# Hyperparameters.
BATCH_SIZE = 4
SEQ_LENGTH = 5
HIDDEN_SIZE = 100

"""
Load and process data, utility functions
"""

# Open and read file.
with open(FILE_NAME,'r') as f:
    raw_data = f.read()

# Get text and vocabulary.
data_size = len(raw_data)
vocab = set(raw_data)
VOCAB_SIZE = len(vocab)
print ('File has %d characters and %d unique.' % (data_size, VOCAB_SIZE))
idx_to_vocab = dict(enumerate(vocab))
vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

# "Embedded" data,
data = [vocab_to_idx[c] for c in raw_data]
# Free memory.
del raw_data

#
#print(p)
#inputs = np.zeros((BATCH_SIZE,  SEQ_LENGTH+1), dtype=int)
#for b in range(BATCH_SIZE):
#    print(b)
#    inputs[b, :] = data[p[b]:p[b]+SEQ_LENGTH+1]
#print (inputs)
#exit(1)

def gen_batch(seq_length, batch_size):
    # Buffer for the sequence of batches.
    batch = np.zeros((batch_size,  seq_length+1), dtype=int)
    # Random "starts of sequence".
    starts = np.random.randint(len(data)-1-(seq_length+1),size=(batch_size)).tolist()
    for b in range(batch_size):
        batch[b, :] = data[starts[b]:starts[b]+seq_length+1]
    return batch


#  for b in range(0,batch_size):
#       p[b] = np.random.randint(len(data)-1-S)
#
#    inputs[:,b] = [char_to_ix[ch] for ch in data[p[b]:p[b]+S]]

def train_network(g, num_iterations, seq_length = SEQ_LENGTH, batch_size = BATCH_SIZE, verbose = True, save=False):
    #tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for iteration in range(num_iterations):
            batch = gen_batch(seq_length, batch_size)
            #training_state = None
            feed_dict={g['input_text']: batch}
            #if training_state is not None:
            #    feed_dict[g['init_state']] = training_state
            training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                  g['final_state'],
                                                  g['train_step']],
                                                         feed_dict)
            if verbose:
                print("Average training loss for Epoch:", training_loss_)
            #training_losses.append(training_loss_)

        #if isinstance(save, str):
        #    g['saver'].save(sess, save)
    # Close session at the end.

    sess.close()
    return training_losses


def build_basic_rnn_graph_with_list(
    hidden_size = HIDDEN_SIZE,
    num_classes = VOCAB_SIZE,
    batch_size = BATCH_SIZE,
    seq_length = SEQ_LENGTH,
    learning_rate = 1e-4):

    # Just in case...
    tf.reset_default_graph()

    #x = tf.placeholder(tf.int32, [batch_size, seq_length], name='input_placeholder')
    #y = tf.placeholder(tf.int32, [batch_size, seq_length], name='labels_placeholder')
    # Create a placeholder for  input text [BATCH_SIZE x SEQ_LENGTH+1] 
    input_text = tf.placeholder(tf.int32, [batch_size, seq_length+1], name='input_placeholder')
    # Slice text into x and y of size [BATCH_SIZE x SEQ_LENGTH] (y is "shifted by 1)
    # Both variables store indices of characters in the "vocabulary"!
    x = tf.slice(input_text,  [0,  0],  [batch_size, seq_length])
    y = tf.slice(input_text,  [0,  1],  [batch_size, seq_length])
    
    # Create tensor with one-hot encoding [BATCH_SIZE x SEQ_LENGTH x INPUT_SIZE]
    x_one_hot = tf.one_hot(x, num_classes, dtype=tf.float32)
    print("x_one_hot =",  x_one_hot.shape)
    
    # Create SEQ_LENGTH list of tensors of size  [BATCH_SIZE x SEQ_LENGTH x INPUT_SIZE]
    # splits = tf.split(x_one_hot, seq_length, 1)
    
    # Create SEQ_LENGTH list of tensors of size  [BATCH_SIZE x INPUT_SIZE]
    rnn_inputs = [tf.squeeze(i,squeeze_dims=[1]) for i in tf.split(x_one_hot, num_or_size_splits=seq_length, axis=1)]
    print("rnn_inputs =", len(rnn_inputs),  " of shape",  rnn_inputs[0].shape)
 
    # Build the RNN.
    cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)
    # RNN outputs: sequenge SEQ_LENGTH of [BATCH_SIZE x HIDDEN_SIZE]

    # Add logits layer.
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [hidden_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    # Logits: sequenge SEQ_LENGTH of [BATCH_SIZE x HIDDEN_SIZE]
    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
    # Stack logits [SEQ_LENGTH x BATCH_SIZE x INPUT_SIZE]
    logits_stack = tf.stack(logits)
    # Reorder logits [BATCH_SIZE x SEQ_LENGTH x INPUT_SIZE] - as required by sequence_loss
    logits_ordered = tf.transpose(logits_stack, [1, 0, 2])
    print("logits_ordered =",   logits_ordered.shape)
    
    loss_weights = tf.ones([batch_size, seq_length])
    # Calculate losses.
    losses = tf.contrib.seq2seq.sequence_loss(logits_ordered, y, loss_weights)
    # Calculate mean loss over sequence and batch.
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    # Return dictionary with "usefull" nodes.
    return dict(
        input_text = input_text,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step
    )

t1 = time.time()
g = build_basic_rnn_graph_with_list()
print("It took", time.time() - t1, "seconds to build the graph.")

t2 = time.time()
train_network(g, 1000)
print("It took", time.time() - t2, "seconds to train for 3 epochs.")
