"""Simple model for char-modelling using LSTMs.
"""

import tensorflow as tf
from models.simpleLSTMCell import SimpleLSTMCell


class SimpleLSTMModel(object):
    """Simple RNN (LSTM) with FC layer on top."""

    def __init__(self,    
            input_size, # Size of input - number of chars in the dictionary.
            hidden_size, # Number of neurons in hidden layer.
            batch_size, # Size of batch.
            seq_length, # Length of input sequence.
            learning_rate = 1e-4):
        # Store variables.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

    def build_inputs(self):
        with tf.name_scope('inputs'):
            # Create a placeholder for  input text [BATCH_SIZE x SEQ_LENGTH+1] 
            self.input_text = tf.placeholder(tf.int32, [self.batch_size, self.seq_length+1], name='input_placeholder')
            # Slice text into x and y of size [BATCH_SIZE x SEQ_LENGTH] (y is "shifted by 1)
            # Both variables store indices of characters in the "vocabulary"!
            x = tf.slice(self.input_text,  [0,  0],  [self.batch_size, self.seq_length],  name="inputs")
            y = tf.slice(self.input_text,  [0,  1],  [self.batch_size, self.seq_length],  name="targets")
            
            # Create tensor with one-hot encoding [BATCH_SIZE x SEQ_LENGTH x INPUT_SIZE]
            x_one_hot_BSI = tf.one_hot(x, self.input_size, dtype=tf.float32,  name="embedded_inputs")
            print("x_one_hot_BSI =",  x_one_hot_BSI.shape)

            # Reorder input - time major [SEQ_LENGTH x BATCH_SIZE x INPUT_SIZE]
            x_one_hot_SBI = tf.transpose(x_one_hot_BSI, [1, 0, 2])
            print("x_one_hot_SBI =",  x_one_hot_SBI.shape)
            # Return x,y
        return x_one_hot_SBI,  y

    def build_inference(self,  x_one_hot_SBI):
        # Build simple RNN.
        cell = SimpleLSTMCell(self.hidden_size)
        init_state = cell.zero_state(self.batch_size, tf.float32)

        # Create dynamic RNN with outputs [BATCH_SIZE x SEQ_LENGTH x HIDDEN_SIZE] (not S x B x H, despite time_major = TRUE !!!)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, x_one_hot_SBI, initial_state=init_state,  time_major=True)
        print("rnn_outputs =",  rnn_outputs.shape)
        print("final_state_c =",  final_state[0].shape)
        print("final_state_h =",  final_state[1].shape)
        
        with tf.name_scope('rnn_outputs'):       
            # List SEQ_LENGTH of [BATCH_SIZE x HIDDEN_SIZE]
            rnn_outputs_S_BI = tf.unstack(rnn_outputs, axis=1)
            print("rnn_outputs_S_BI =", len(rnn_outputs_S_BI),  " of shape",  rnn_outputs_S_BI[0].shape)

        # Add logits layer.
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [self.hidden_size, self.input_size])
            b = tf.get_variable('b', [self.input_size], initializer=tf.constant_initializer(0.0))

        with tf.name_scope('outputs'):
            # Logits: sequenge SEQ_LENGTH of [BATCH_SIZE x HIDDEN_SIZE]
            logits_S_BI = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs_S_BI]
            print("logits_S_BI =", len(logits_S_BI),  " of shape",  logits_S_BI[0].shape)
            
            # Stack logits [SEQ_LENGTH x BATCH_SIZE x INPUT_SIZE]
            logits_SBI = tf.stack(logits_S_BI)
          
        return logits_SBI,  final_state
    
    def build_loss(self, logits_SBI,  y):
        with tf.name_scope('loss'):
            loss_weights = tf.ones([self.batch_size, self.seq_length])
            # Calculate losses.
            losses = tf.contrib.seq2seq.sequence_loss(logits_SBI, y, loss_weights)
            # Calculate mean loss over sequence and batch.
            total_loss = tf.reduce_mean(losses)
            # Add loss summary.
            tf.summary.scalar("loss", total_loss)
        return total_loss
    
    def build_train(self, total_loss):
        with tf.name_scope('optimizer'):
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(total_loss)
        return train_step

    def core_builder(self):
        # Get x & y.
        x_one_hot_SBI,  y = self.build_inputs()
         
        # Build inference engine.
        logits_SBI,  final_state = self.build_inference(x_one_hot_SBI)
        
        # Add loss.
        total_loss = self.build_loss(logits_SBI,  y)
        
        train_step = self.build_train(total_loss)
        
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        
        # Return dictionary with "usefull" nodes.
        return dict(
            input_text = self.input_text,
            #init_state = init_state,
            final_state = final_state,
            total_loss = total_loss,
            train_step = train_step, 
            merged_summary_op = merged_summary_op
        )
          
    #def one_step(self, sess, input_seq):
    #    outputs = [self.loss, self.gradient_ops]
    #    return sess.run(outputs, feed_dict={self.input_text: input_seq})

