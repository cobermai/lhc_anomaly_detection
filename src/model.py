import math
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans


class Model:
    """
    Model agnostic explainer class.
    """
    def __init__(self,
                 model: object,
                 input_shape: tuple,
                 output_directory: Path,
                 epochs: int,
                 batch_size: int,
                 latent_dim: int,
                 decoder_only: bool = False
                 ):
        """
        Initializes the explainer with specified settings
        """
        super(Model, self).__init__()
        self.input_shape = input_shape
        self.output_directory = output_directory
        self.epochs = epochs
        self.batch_size = batch_size
        self.decoder_only = decoder_only
        if decoder_only:
            self.model = model.Decoder(original_dim=input_shape)
        else:
            self.model = model.AutoEncoder(
                original_dim=input_shape,
                latent_dim=latent_dim)

    def fit_model(self, X, context=None):
        """
        function fits model-agnostic explainer
        :param X: data
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        mse_loss = tf.keras.losses.MeanSquaredError()
        loss_metric = tf.keras.metrics.Mean()

        bat_per_epoch = math.floor(len(X) / self.batch_size)
        # Iterate over epochs.
        for epoch in range(self.epochs):
            print("Start of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step in range(bat_per_epoch):

                n = step * self.batch_size
                # generate training batch
                x_batch_train = X[n:n + self.batch_size].astype(np.float32)
                if self.decoder_only:
                    context_batch_train = context[n:n + self.batch_size].astype(np.float32)
                with tf.GradientTape() as tape:
                    if self.decoder_only:
                        x_reconstructed = self.model(context_batch_train)
                    else:
                        latent = self.model.encoder(x_batch_train)
                        x_reconstructed = self.model.decoder(latent)
                    # Loss
                    mse_loss_x = mse_loss(x_batch_train, x_reconstructed)
                    loss = mse_loss_x

                grads = tape.gradient(loss, self.model.trainable_weights)
                optimizer.apply_gradients(
                    zip(grads, self.model.trainable_weights))

                loss_metric(loss)
                if step % 100 == 0:
                    print(f"loss:{loss.numpy()}, "
                          f"mse_loss_x:{mse_loss_x.numpy()}, ")  # ,f"mse_loss_latent:{mse_loss_latent.numpy()}")

                pd.DataFrame({"epoch": epoch}, index=[0]).to_csv(
                    self.output_directory / "epoch.csv")

#        self.model.save_weights(
         #   str(self.output_directory / "ae_weights.h5"))

    def get_concepts_kmeans(self, X):
        """
        :param X: data
        :return: reconstructed concepts and their lower dimensional prototypes
        """
        latent = self.model.encoder(X)

        kmeans = KMeans(n_clusters=4, random_state=0).fit(np.nan_to_num(latent))
        centers = kmeans.cluster_centers_

        reconstructed = np.zeros(tuple((4,)) + np.shape(X[0]))
        for i, center in enumerate(centers):
            #print(f"values of kmeans center {i}: {center}")
            reconstructed[i] = self.model.decoder(np.array([center]))
        return reconstructed, centers

    #def get_reconstructions(self, X):
