# Python
import tensorflow as tf
sess = tf.Session()

# Create a linear neuron model.
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# Add MSE loss function.
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# Add optimization function - simple SGD.
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Training data.
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# Reset values to "incorrect" defaults.
init = tf.global_variables_initializer()
sess.run(init)

# Log writer
#writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

# Training.
print("Training the model. Please wait...")
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

print(sess.run([W, b]))

# Evaluate the training accuracy.
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

