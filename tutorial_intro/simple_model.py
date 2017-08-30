# Python
import tensorflow as tf
sess = tf.Session()

# Create a linear neuron model.
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# Initialize variables.
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x:[1,2,3,4]}))

# Add MSE loss function.
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# Run and check loss.
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# Assign "perfect" values of W and b.
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])

# Run and check loss.
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

