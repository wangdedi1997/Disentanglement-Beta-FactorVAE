import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os

# 0. Settings
# ------------------------------------------------------------------------------
# By default, we save all the results in subdirectories of the following path.
base_path = "example_output"

# Training parameters
iter_num = 1
my_training_steps = 1000
my_batch_size = 64
my_z_dims = 2
my_neuron_num = 128
my_learning_rate1 = [1e-4]
my_learning_rate2 = 1e-4
my_beta = [1]
mygamma = [1]

# 1. Import the toy dataset
# ------------------------------------------------------------------------------

# Import the data
# dataset_factors = np.loadtxt('toy_dataset_factors1')
# dataset_representations = np.loadtxt('toy_dataset_representations1')

# dataset_factors = np.loadtxt('toy_dataset_factors2')
dataset_representations = np.loadtxt('toy_dataset_representations2')

# 2. Train beta_factor_VAE.
# ------------------------------------------------------------------------------

# We save the results in a `factorVAE` subfolder.
path_factor_vae = os.path.join(base_path, "beta_factor_VAE")

class Beta_Factor_VAE(object):
    def __init__(self, z_dim, dataset, path, beta=1.0, gamma=0.0, neuron_num=128, learning_rate1=1e-4,
                 learning_rate2=1e-4):
        self.z_dim = z_dim
        self.gamma = gamma
        self.path = path + "_beta=%.2f_gamma=%.1f" % (beta, gamma)
        self.neuron_num = neuron_num
        self.learning_rate1 = learning_rate1
        self.learning_rate2 = learning_rate2
        self.beta = beta

        self.data_shape, self.data_train, self.data_test = self._data_init(dataset)
        self.input_tensor, self.z_mean, self.z_logvar, self.z_sample, self.reconstructions = self._autoencoder_init()
        self.reconstruction_loss, self.auto_encoder_loss, self.disc_loss, self.tc_loss, self.loss = self._loss_init()
        self.ae_train_step, self.disc_train_step = self._optimizer_init()

        self.sess = tf.Session()
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
        self.sess.run(tf.global_variables_initializer())

    def _data_init(self, dataset):
        # data shape
        data_shape = dataset.shape[1:]

        # 90% random test/train split
        n_data = len(dataset)
        np.random.shuffle(dataset)
        data_train = dataset[0: (9 * n_data) // 10]
        data_test = dataset[(9 * n_data) // 10:]

        return data_shape, data_train, data_test

    def _autoencoder_init(self):
        # make placeholder for feeding in data during training and evaluation
        input_tensor = tf.placeholder(shape=[None] + list(self.data_shape), dtype=tf.float32, name="input")
        # define the encoder network
        z_mean, z_logvar = self._encoder_init(input_tensor)
        # reparameterization trick
        eps = tf.random_normal(shape=tf.shape(z_mean))
        z_sample = z_mean + (tf.exp(z_logvar / 2) * eps)
        # define decoder network. reconstructions is decoding of z sample
        reconstructions = self._decoder_init(z_sample)

        return input_tensor, z_mean, z_logvar, z_sample, reconstructions

    def _encoder_init(self, inputs):

        with tf.variable_scope("encoder"):
            flattened = tf.layers.flatten(inputs)
            # linear encoder for mean
            means = tf.layers.dense(flattened, self.z_dim, activation=None, name="means")

            # MLP encoder for log_var
            e1 = tf.layers.dense(flattened, self.neuron_num, activation=tf.nn.relu)
            e2 = tf.layers.dense(e1, self.neuron_num, activation=tf.nn.relu)
            log_var = tf.layers.dense(e2, self.z_dim, activation=None, name="log_var")

        return means, log_var

    def _decoder_init(self, latent_tensor):
        # MLP decoder
        with tf.variable_scope("decoder"):
            d1 = tf.layers.dense(latent_tensor, self.neuron_num, activation=tf.nn.tanh)
            d2 = tf.layers.dense(d1, self.neuron_num, activation=tf.nn.tanh)
            d3 = tf.layers.dense(d2, self.neuron_num, activation=tf.nn.tanh)
            d4 = tf.layers.dense(d3, np.prod(self.data_shape))

        return tf.reshape(d4, shape=[-1] + list(self.data_shape))

    def _loss_init(self):
        ### Regulariser part of loss has two parts: KL divergence and Total Correlation
        ## KL part:
        kl_loss = tf.reduce_mean(
            0.5 * tf.reduce_sum(
                tf.square(self.z_mean) + tf.exp(self.z_logvar) - self.z_logvar - 1, [1]),
            name="kl_loss")

        ## Total Correlation part:
        # permuted samples from q(z)
        real_samples = self.z_sample
        permuted_rows = []
        for i in range(real_samples.get_shape()[1]):
            permuted_rows.append(tf.random_shuffle(real_samples[:, i]))
        permuted_samples = tf.stack(permuted_rows, axis=1)

        # define discriminator network to distinguish between real and permuted q(z)
        logits_real, probs_real = self._discriminator_init(real_samples)
        logits_permuted, probs_permuted = self._discriminator_init(permuted_samples, reuse=True)

        # FactorVAE paper has gamma * log(D(z) / (1- D(z))) in Algorithm 2, where D(z) is probability of being real
        tc_loss = tf.reduce_mean(logits_real[:, 0] - logits_real[:, 1], axis=0)

        total_regulariser = self.beta * kl_loss + self.gamma * tc_loss

        ### Reconstruction loss is (x-x')^2/2/sigma^2, sigma=0.01
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
            tf.square(self.input_tensor - self.reconstructions) / 2 / 0.01 / 0.01, [1]), name="reconstruction_loss")

        ### Evidence Lower Bound (ELBO)
        auto_encoder_loss = tf.add(reconstruction_loss, kl_loss, name="auto_encoder_loss")

        ### training object function
        loss = tf.add(reconstruction_loss, total_regulariser, name="auto_encoder_loss")

        ### Loss for discriminator
        disc_loss = tf.add(0.5 * tf.reduce_mean(tf.log(probs_real[:, 0])),
                           0.5 * tf.reduce_mean(tf.log(probs_permuted[:, 1])), name="discriminator_loss")

        return reconstruction_loss, auto_encoder_loss, disc_loss, tc_loss, loss

    def _discriminator_init(self, inputs, reuse=False):
        with tf.variable_scope("discriminator"):
            flattened = tf.layers.flatten(inputs)
            d1 = tf.layers.dense(flattened, 128, activation=tf.nn.leaky_relu, name="d1", reuse=reuse)
            d2 = tf.layers.dense(d1, 128, activation=tf.nn.leaky_relu, name="d2", reuse=reuse)
            logits = tf.layers.dense(d2, 2, activation=None, name="logits", reuse=reuse)
            probs = tf.nn.softmax(logits)

        return logits, probs

    def _optimizer_init(self):

        enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        ae_train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate1).minimize(self.loss,
                                                                                           var_list=enc_vars + dec_vars)
        disc_train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate2, beta1=0.5, beta2=0.9).minimize(
            -self.disc_loss, var_list=disc_vars)

        return ae_train_step, disc_train_step

    def train(self, training_steps, batch_size):
        log_path = self.path + '_train.log'
        start = time.time()

        print("Beginning training")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        print("Beginning training", file=open(log_path, 'w'))
        it = 0
        while it < training_steps:
            it += 1
            self.sess.run(self.ae_train_step, {self.input_tensor: self.sample_minibatch(batch_size=batch_size)})
            self.sess.run(self.disc_train_step, {self.input_tensor: self.sample_minibatch(batch_size=batch_size)})

            if it % 500 == 0:
                batch = self.sample_minibatch(batch_size=batch_size)
                ae_train_loss = self.sess.run(self.auto_encoder_loss, {self.input_tensor: batch})
                recon_train_loss = self.sess.run(self.reconstruction_loss, {self.input_tensor: batch})
                disc_train_loss = self.sess.run(self.disc_loss, {self.input_tensor: batch})
                train_loss = self.sess.run(self.loss, {self.input_tensor: batch})

                train_time = time.time() - start

                print(
                    "Iteration %i:\tTime %f s\nAutoencoder loss (train) %f\tReconstruction loss (train) %f\t"
                    "Loss (train) %f\tDiscriminator loss (train) %f" % (
                        it, train_time, ae_train_loss, recon_train_loss, train_loss, disc_train_loss), flush=True)
                print(
                    "Iteration %i:\tTime %f s\nAutoencoder loss (train) %f\tReconstruction loss (train) %f\t"
                    "Loss (train) %f\tDiscriminator loss (train) %f" % (
                        it, train_time, ae_train_loss, recon_train_loss, train_loss, disc_train_loss), flush=True,
                    file=open(log_path, 'a'))

                batch = self.sample_minibatch(batch_size=batch_size, test=True)
                ae_test_loss = self.sess.run(self.auto_encoder_loss, {self.input_tensor: batch})
                recon_test_loss = self.sess.run(self.reconstruction_loss, {self.input_tensor: batch})
                disc_test_loss = self.sess.run(self.disc_loss, {self.input_tensor: batch})
                test_loss = self.sess.run(self.loss, {self.input_tensor: batch})
                print(
                    "Autoencoder loss (test) %f\tReconstruction loss (test) %f\tDiscriminator loss (test) %f\tLoss %f"
                    % (ae_test_loss, recon_test_loss, disc_test_loss, test_loss), flush=True)
                print(
                    "Autoencoder loss (test) %f\tReconstruction loss (test) %f\tDiscriminator loss (test) %f\tLoss %f"
                    % (ae_test_loss, recon_test_loss, disc_test_loss, test_loss), flush=True, file=open(log_path, 'a'))

            if it % 10000 == 0:
                model_path = self.path + "checkpoints/model"
                save_path = self.saver.save(self.sess, model_path, global_step=it)
                print("Model saved to: %s" % save_path)
                print("Model saved to: %s" % save_path, file=open(log_path, 'a'))

        # output the saving path
        total_training_time = time.time() - start
        print("Total training time: %f" % total_training_time)
        print("Total training time: %f" % total_training_time, file=open(log_path, 'a'))
        model_path = self.path + "checkpoints/model"
        save_path = self.saver.save(self.sess, model_path, global_step=it)
        print("Model saved to: %s" % save_path)
        print("Model saved to: %s" % save_path, file=open(log_path, 'a'))

        # output the final encoder model
        print(self.sess.run('encoder/means/kernel:0'), file=open(log_path, 'a'))
        print(self.sess.run('encoder/means/bias:0'), file=open(log_path, 'a'))

    def load_latest_checkpoint(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.path + 'checkpoints'))

    def sample_minibatch(self, batch_size=64, test=False):
        if test is False:
            indices = np.random.choice(range(len(self.data_train)), batch_size, replace=False)
            sample = self.data_train[indices]
        else:
            indices = np.random.choice(range(len(self.data_test)), batch_size, replace=False)
            sample = self.data_test[indices]
        return sample

    def output_final_result(self):
        # output the result
        ae_train_loss = self.sess.run(self.auto_encoder_loss, {self.input_tensor: self.data_train})
        recon_train_loss = self.sess.run(self.reconstruction_loss, {self.input_tensor: self.data_train})
        disc_train_loss = self.sess.run(self.disc_loss, {self.input_tensor: self.data_train})
        train_loss = self.sess.run(self.loss, {self.input_tensor: self.data_train})

        ae_test_loss = self.sess.run(self.auto_encoder_loss, {self.input_tensor: self.data_test})
        recon_test_loss = self.sess.run(self.reconstruction_loss, {self.input_tensor: self.data_test})
        disc_test_loss = self.sess.run(self.disc_loss, {self.input_tensor: self.data_test})
        test_loss = self.sess.run(self.loss, {self.input_tensor: self.data_test})

        train_tc = self.sess.run(self.tc_loss, {self.input_tensor: self.data_train})
        test_tc = self.sess.run(self.tc_loss, {self.input_tensor: self.data_test})

        return [self.beta, self.gamma, ae_train_loss, recon_train_loss, train_loss, disc_train_loss, train_tc,
                ae_test_loss, recon_test_loss, test_loss, disc_test_loss, test_tc]

    def latent_representation(self):
        test_path = self.path + '_x_test.csv'
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        print("# Test Set", file=open(test_path, 'w'))

        representation_path = self.path + '_representation.csv'
        os.makedirs(os.path.dirname(representation_path), exist_ok=True)
        print("# Latent Representation of Test Set", file=open(representation_path, 'w'))

        mean_representation_path = self.path + '_mean_representation.csv'
        os.makedirs(os.path.dirname(mean_representation_path), exist_ok=True)
        print("# Latent Representation of Test Set", file=open(representation_path, 'w'))

        # get the latent representation of test set
        z_sample = self.sess.run(self.z_sample, {self.input_tensor: self.data_test})
        z_mean = self.sess.run(self.z_mean, {self.input_tensor: self.data_test})

        x_test_df = pd.DataFrame(self.data_test)
        z_sample_df = pd.DataFrame(z_sample)
        z_mean_df = pd.DataFrame(z_mean)

        # save to csv
        x_test_df.to_csv(test_path, index=False, header=True)
        z_sample_df.to_csv(representation_path, index=False, header=True)
        z_mean_df.to_csv(mean_representation_path, index=False, header=True)


    def latent_traversal(self):
        traversal_path = self.path + '_traversal.csv'
        os.makedirs(os.path.dirname(traversal_path), exist_ok=True)
        print("# Latent Trasversal", file=open(traversal_path, 'w'))

        # samples from Gaussian distribution
        factors = np.random.normal(loc=0, scale=1, size=(len(self.data_test), self.z_dim))

        # get reconstruction
        x_reconstructions = self.sess.run(self.reconstructions, {self.z_sample: factors})
        x_reconstructions_df = pd.DataFrame(x_reconstructions)

        # save to csv
        x_reconstructions_df.to_csv(traversal_path, index=False, header=True)

# training the factorVAE
output_final_result_path = path_factor_vae + '_final_result.csv'
output_final_result_mean_path = path_factor_vae + '_final_result_mean.csv'
os.makedirs(os.path.dirname(output_final_result_path), exist_ok=True)
os.makedirs(os.path.dirname(output_final_result_mean_path), exist_ok=True)

# record the final result
result = {"beta": [], "gamma": [], "ae_train_loss": [], "recon_train_loss": [], "train_loss": [],
          "disc_train_loss": [], "train_tc": [], "ae_test_loss": [], "recon_test_loss": [], "test_loss": [],
          "disc_test_loss": [], "test_tc": []}
final_result = pd.DataFrame(result)
final_result_mean = pd.DataFrame(result)

min_ELBO = 1e7

for beta_sample in my_beta:
    for gamma_sample in mygamma:
        for learning_rate1 in my_learning_rate1:
            for iter in range(iter_num):
                # Random seed
                np.random.seed(iter)
                tf.set_random_seed(iter)

                # reset tensorflow
                tf.reset_default_graph()
                myfactorVAE = Beta_Factor_VAE(my_z_dims, dataset_representations, path_factor_vae, beta_sample,
                                              gamma_sample, my_neuron_num, learning_rate1, my_learning_rate2)
                myfactorVAE.train(my_training_steps, my_batch_size)
                result_sample = myfactorVAE.output_final_result()

                final_result.loc[len(final_result)] = result_sample

                if min_ELBO > final_result.loc[len(final_result) - 1]["ae_test_loss"]:
                    myfactorVAE.latent_representation()
                    myfactorVAE.latent_traversal()

            # output the mean value
            final_result_mean = final_result_mean.append(
                final_result.loc[len(final_result) - iter_num:len(final_result)].mean(axis=0), ignore_index=True)

final_result.to_csv(output_final_result_path, index=True, header=True)
final_result_mean.to_csv(output_final_result_mean_path, index=True, header=True)
