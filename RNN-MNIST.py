import tensorflow as tf
from utils import variable_summaries
from tqdm import tqdm
# Import MNIST Dataset
from tensorflow.examples.tutorials import mnist

mnist = mnist.input_data.read_data_sets("/tmp/data/", one_hot=True)

# Define model parameters
element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

# Where to save tensorBoard Model summaries
LOG_DIR = "logs/RNN_with_summaries"

# Create placeholders for inputs, labels
# shape is (batch_size, RNN iterations, element size)
# The number of iterations is the width of the image and
# it is fixed totally for implementation optimization purposes.
# element size is the height of the image.
_inputs = tf.placeholder(dtype=tf.float32,
                         shape=[None, time_steps, element_size],
                         name="inputs")

y = tf.placeholder(dtype=tf.float32,
                   shape=[None, 10],
                   name="labels")

# Weight and bias for input and hidden layer
with tf.name_scope("rnn_weights"):
    with tf.name_scope("W_x"):
        Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
        variable_summaries(Wx)
    with tf.name_scope("W_h"):
        Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
        variable_summaries(Wh)
    with tf.name_scope("Bias"):
        b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
        variable_summaries(b_rnn)


# Run one RNN model step
def rnn_step(previous_hidden_state, x):
    current_hidden_state = tf.tanh(tf.matmul(previous_hidden_state, Wh) + tf.matmul(x, Wx)+b_rnn)
    return current_hidden_state

# Processing input to work with scan function
# Current input shape: (batch_size, time_steps, element_size)
processed_input = tf.transpose(_inputs, perm=[1, 0, 2])
# Current input shape: (time_steps, batch_size, element_size)

initial_hidden = tf.zeros([batch_size, hidden_layer_size])

# Getting all state vectors across time
all_hidden_states = tf.scan(fn=rnn_step,
                            elems=processed_input,
                            initializer=initial_hidden,
                            name="states")

# Weights for output layer
with tf.name_scope("linear_layer_weights") as scope:
    with tf.name_scope("W_linear"):
        Wl = tf.Variable(tf.truncated_normal(shape=[hidden_layer_size, num_classes], mean=0, stddev=.01))
        variable_summaries(Wl)
    with tf.name_scope("Bias_linear"):
        bl = tf.Variable(tf.truncated_normal(shape=[num_classes], mean=0, stddev=.01))
        variable_summaries(bl)


# Apply linear layer to state vector
def get_linear_layer(hidden_state):
    return tf.matmul(hidden_state, Wl) + bl

with tf.name_scope("linear_layer_weights") as scope:
    # Iterate through time and apply linear layer to all RNN outputs
    all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)
    # Get last ouput
    output = all_outputs[-1]
    tf.summary.histogram("outputs", output)

with tf.name_scope("cross_entropy"):
    cross_entropy_all = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
    cross_entropy = tf.reduce_mean(cross_entropy_all)
    tf.summary.scalar("cross_entropy", cross_entropy)

with tf.name_scope("train"):
    # Using RMSPropOptimizer
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
    tf.summary.scalar("accuracy", accuracy)

# Merge all summaries
merged = tf.summary.merge_all()

# Get a small test set
test_data = mnist.test.images[:batch_size].reshape((-1, time_steps, element_size))
test_label = mnist.test.labels[:batch_size]

with tf.Session() as sess:
    # Write summaries to LOG_DIR -- used by TensorBoard
    train_writer = tf.summary.FileWriter(LOG_DIR + "/train",
                                         graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(LOG_DIR + "/test",
                                        graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())

    for i in tqdm(range(10001)):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 sequences of 28 pixels
        batch_x = batch_x.reshape((batch_size, time_steps, element_size))
        summary, _ = sess.run(fetches=[merged, train_step],
                              feed_dict={_inputs: batch_x, y: batch_y})
        # Add to summaries
        train_writer.add_summary(summary, i)

        if i % 1000 == 0 :
            acc, loss, = sess.run([accuracy, cross_entropy],
                                  feed_dict={_inputs: batch_x, y: batch_y})
            print("Iter " + str(i) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

        if i % 10 == 0:
            # calculate accuracy for the 128 MNIST test images and add to summaries
            summary, acc = sess.run([merged, accuracy],
                                    feed_dict={_inputs: test_data, y: test_label})
            test_writer.add_summary(summary, i)
    test_acc = sess.run(accuracy,
                        feed_dict={_inputs: test_data, y: test_label})
    print("Test Accuracy:", test_acc)
