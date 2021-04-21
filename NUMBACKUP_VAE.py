import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# A layer that takes mean/variance as inputs and returns a random sample z
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Takes manifest variables; outputs mean/logvar for each latent distribution
def create_encoder(manifest_dim, hidden_dim, latent_dim):
    # Encoder
    encoder_inputs = keras.Input(shape=(manifest_dim,))
    encoder_x = layers.Dense(hidden_dim, activation="tanh")(encoder_inputs)
    encoder_x = layers.Dense(hidden_dim, activation="tanh")(encoder_x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(encoder_x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(encoder_x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

# Takes value for each latent variable; outputs the manifest variables
def create_decoder(manifest_dim, hidden_dim, latent_dim, cat_lengths, cat_idx, num_hh_controls):
    # Decoder
    decoder_latent_inputs = keras.Input(shape=(latent_dim,))
    decoder_control_inputs = keras.Input(shape=(num_hh_controls,))
    decoder_inputs = keras.layers.Concatenate(axis=1)([decoder_latent_inputs, decoder_control_inputs])
    decoder_x = layers.Dense(hidden_dim, activation="tanh")(decoder_inputs)
    decoder_x = layers.Dense(hidden_dim, activation="tanh")(decoder_x)
#     decoder_num_outputs = layers.Dense(cat_idx, activation="sigmoid")(decoder_x)
    decoder_cat_outputs = [layers.Dense(var_length, activation="softmax")(decoder_x) for var_length in cat_lengths]
    decoder = keras.Model([decoder_latent_inputs, decoder_control_inputs], [decoder_cat_outputs], name="decoder")
    return decoder

# Define custom loss function for combined numerical and categorical data
def get_reconstruction_loss(data, reconstruction, cat_idx, cat_lengths):
#     # Handle case with no categorical variables
#     if len(cat_lengths) > 0:
#         reconstruction_num = reconstruction[0]
#     else:
#         reconstruction_num = reconstruction

#     # Mean squared error for numerical variables
#     data_num = data[:,:cat_idx]
#     loss_num = keras.losses.mean_squared_error(data_num, reconstruction_num)
#     loss_num = tf.reduce_sum(loss_num)  # Add the avg losses for each variable in current epoch

    # Categorical cross entropy for categorical variables
    loss_list = []
    current = cat_idx
    for i, x in enumerate(cat_lengths):
        data_cat = data[:,current:(current + x)]
#         reconstruction_cat = reconstruction[1][i]
        reconstruction_cat = reconstruction[0][i]
        loss = keras.losses.categorical_crossentropy(data_cat, reconstruction_cat, from_logits=False)
        loss = tf.reduce_mean(loss)  # Average the loss over the batch for current variable
        loss_list.append(loss)
        current += x
    loss_cat = tf.reduce_sum(loss_list)  # Add the avg losses for each variable in current epoch

    # Return both losses; they are combined in the model
    return (loss_cat)

# Variational Autoencoder
class VAE(keras.Model):
    def __init__(self, manifest_dim, hidden_dim, latent_dim, cat_lengths, cat_idx, num_hh_controls, kl_weight, num_weight, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = create_encoder(manifest_dim, hidden_dim, latent_dim)
        self.decoder = create_decoder(manifest_dim, hidden_dim, latent_dim, cat_lengths, cat_idx, num_hh_controls)
        self.manifest_dim = manifest_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cat_lengths = cat_lengths
        self.cat_idx = cat_idx
        self.num_hh_controls = num_hh_controls
        self.kl_weight = kl_weight
        self.num_weight = num_weight
        self.tot_loss_tracker = keras.metrics.Mean(name="tot_loss")
        self.rec_loss_tracker = keras.metrics.Mean(name="rec_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            # Get latent vars from the encoder; feed to decoder and get sampled manifest variables
            z_mean, z_log_var, z = self.encoder(data[:,:self.manifest_dim])
            reconstruction = self.decoder([z, data[:,self.manifest_dim:]])

            # Get loss between input values and reconstruction
            loss_cat = get_reconstruction_loss(
                data,
                reconstruction,
                self.cat_idx,
                self.cat_lengths
            )
#             loss_num *= self.num_weight
#             reconstruction_loss = loss_num + loss_cat
            reconstruction_loss = loss_cat

            # Get Kullback Leibler loss between normal distribution and actual for latent variables
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = self.kl_weight * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # Combine into single loss term
            tot_loss = reconstruction_loss + kl_loss

        # Get new gradients given the loss and take another step (update weights)
        grads = tape.gradient(tot_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Record the loss metrics
        self.tot_loss_tracker.update_state(tot_loss)
        self.rec_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "tot_loss": self.tot_loss_tracker.result(),
            "rec_loss": self.rec_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs[:,:self.manifest_dim])
        reconstruction = self.decoder([z, inputs[:,self.manifest_dim:]])
        return reconstruction

    @property
    def metrics(self):
        return [self.tot_loss_tracker, self.rec_loss_tracker, self.kl_loss_tracker]