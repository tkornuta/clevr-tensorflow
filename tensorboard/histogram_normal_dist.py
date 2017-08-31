# Example from
# https://stackoverflow.com/questions/35567132/meaning-of-histogram-on-tensorboard
import tensorflow as tf

log_dir = "/tmp/tf/test_logs"

# Eventually clear the log dir.
if tf.gfile.Exists(log_dir):
  tf.gfile.DeleteRecursively(log_dir)
# Create (new) log dir.
tf.gfile.MakeDirs(log_dir)

# Generate variables with normal distribution.
W1 = tf.Variable(tf.random_normal([200, 10], stddev=1.0))
W2 = tf.Variable(tf.random_normal([200, 10], stddev=0.13))

# Create TF histograms.
w1_hist = tf.summary.histogram("weights-stdev 1.0", W1)
w2_hist = tf.summary.histogram("weights-stdev 0.13", W2)

# Merge summary.
summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()
session = tf.Session()

# Create a TF writer.
writer = tf.summary.FileWriter(log_dir, session.graph)

session.run(init)

for i in xrange(2):
    writer.add_summary(sess.run(summary_op),i)

writer.flush()
writer.close()
session.close()


