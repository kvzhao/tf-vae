from VAE import VariationalAutoencoder
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples

learning_rate = 0.001
batch_size = 100
training_epochs = 10
display_step = 5

network_architecture = \
	dict(n_hidden_recog_1 = 500,
		 n_hidden_recog_2 = 500,
		 n_hidden_gener_1 = 500,
		 n_hidden_gener_2 = 500,
		 n_input = 784,
		 n_z = 20
		 )

vae = VariationalAutoencoder(network_architecture, 
							learning_rate=learning_rate, 
							batch_size=batch_size)
# Training cycle
for epoch in range(training_epochs):
	avg_cost = 0.
	total_batch = int(n_samples / batch_size)
	# Loop over all batches
	for i in range(total_batch):
		batch_xs, _ = mnist.train.next_batch(batch_size)
            # Fit training using batch data
		cost = vae.partial_fit(batch_xs)
	# Compute average loss
	avg_cost += cost / n_samples * batch_size
	# Display logs per epoch step
	if epoch % display_step == 0:
		print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)