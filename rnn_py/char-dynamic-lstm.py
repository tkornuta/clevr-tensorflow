import time
import numpy as np

import tensorflow as tf
from models.simpleLSTMModel import SimpleLSTMModel

"""
Load and process data, utility functions
"""
def load_data(file_name):
    # Open and read file.
    with open(FILE_NAME,'r') as f:
        raw_data = f.read()

    # Get text and vocabulary.
    data_size = len(raw_data)
    vocab = set(raw_data)
    vocab_size = len(vocab)
    print ('File has %d characters and %d unique.' % (data_size, vocab_size))
    idx_to_vocab = dict(enumerate(vocab))
    vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

    # "Embedded" data,
    data = [vocab_to_idx[c] for c in raw_data]
    # Free memory.
    del raw_data

    return [data,  data_size, vocab_size,  idx_to_vocab, vocab_to_idx]



def gen_batch(data,  batch_size,  seq_length):
    # Buffer for the sequence of batches.
    batch = np.zeros((batch_size,  seq_length+1), dtype=int)
    # Random "starts of sequence".
    starts = np.random.randint(len(data)-1-(seq_length+1),size=(batch_size)).tolist()
    for b in range(batch_size):
        batch[b, :] = data[starts[b]:starts[b]+seq_length+1]
    return batch

def train_network(g, num_iterations, summary_frequency,  batch_size,  seq_length):
    #tf.set_random_seed(2345)
    with tf.Session() as sess:
        # Initialize variables.
        sess.run(tf.global_variables_initializer())
        # Create summary writers, point them to LOG_DIR.
        train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)

        # Perform #num_iterations learning iterations.
        for step in range(np.int(num_iterations)):
            batch = gen_batch(data, batch_size,  seq_length)
            #training_state = None
            feed_dict={g['input_text']: batch}
            #if training_state is not None:
            #    feed_dict[g['init_state']] = training_state
            training_loss_, training_state, _,  summary_ = sess.run([g['total_loss'],
                                                  g['final_state'],
                                                  g['train_step'], 
                                                  g['merged_summary_op']],
                                                         feed_dict)
                                                         
             # Every x steps collect statistics.
            if step % summary_frequency == 0:
                train_writer.add_summary(summary_, step*seq_length*batch_size)
                print("Average training loss at iteration ", step,  " (", step*seq_length*batch_size, "): ",   training_loss_)
            
        #if isinstance(save, str):
        #    g['saver'].save(sess, save)
 
    # Close writer and session.
    train_writer.close()
    sess.close()


###########################
if __name__ == "__main__":
    # Hyperparameters.
    BATCH_SIZE = 50
    SEQ_LENGTH = 20
    HIDDEN_SIZE = 100

    # Dirs - must be absolute paths!
    #LOG_DIR = '/tmp/tf/char-dynamic-lstm/'
    LOG_DIR = "/tmp/tf/char-dynamic-lstm/B"+\
        str(BATCH_SIZE)+"S"+str(SEQ_LENGTH)+"_H"+str(HIDDEN_SIZE)+"/"
    print("Writing TB log to:",LOG_DIR)
    
    # Text file.
    #FILE_NAME = "/home/tkornuta/data/tiny-shakespeare/tiny-shakespeare.txt"
    FILE_NAME = "/home/tkornuta/data/ptb/ptb.train.txt"

    # Eventually clear the log dir.
    if tf.gfile.Exists(LOG_DIR):
      tf.gfile.DeleteRecursively(LOG_DIR)
    # Create (new) log dir.
    tf.gfile.MakeDirs(LOG_DIR)

    [data,  data_size, vocab_size, idx_to_vocab, vocab_to_idx] = load_data(FILE_NAME)

    # Building model.
    t1 = time.time()
    model = SimpleLSTMModel(vocab_size, HIDDEN_SIZE, BATCH_SIZE,  SEQ_LENGTH)
    g = model.core_builder()
    print("It took", time.time() - t1, "seconds to build the graph.")

    # Determine how long to perform the training and how often the test loss on validation batch will be computed. 
    summary_frequency = 100
    num_steps = 1e4 

    t2 = time.time()
    train_network(g, num_steps,  summary_frequency, BATCH_SIZE,  SEQ_LENGTH)
    print("It took", time.time() - t2, "seconds to train.")


