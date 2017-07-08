import tensorflow as tf

#Nodes for tensor
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2) #Gives the tensor object, not the value of the tensor

# Nodes run on sessions
sess = tf.Session()

print(sess.run([node1, node2])) #Nodes running on session

# Nodes can be used to perform actions and make computational graph
node3 = tf.add(node1, node2)
print(sess.run(node3))

#Placeholders return a value (Dynamic)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1,3], b: [4.5,8]}))

print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))

# Variables allow us to add trainable parameters to a graph
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b # A simple linear model y = m * x + c

# Variables are not initialized when you call tf.Variable
# It needs an initialzer to intialze the variable
init = tf.global_variables_initializer() 
sess.run(init)

print(sess.run(linear_model, {x:[1,2,3,4]}))

# A loss function measures how far apart the current model is from the provided data
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# tf.assign can be used to give new values to the Variables
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# Optimizer reduces the loss function to minimum. We have used gradient descent here
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))