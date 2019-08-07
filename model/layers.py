from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class Encoder(tf.keras.layers.Layer):

    def __init__(self, thought_size, word_size, vocab_size, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.thought_size = thought_size
        self.word_size = word_size
        self.vocab_size = vocab_size
        self.embed = tf.keras.layers.Embedding(self.vocab_size, self.word_size)
        self.bigru = tf.keras.layers.Bidirectional(
            tf.compat.v1.keras.layers.CuDNNGRU(self.thought_size, return_sequences=False, return_state=True)
        )

    def call(self, inputs, training=None):
        # sentence = [None, 30] with zero post padded
        x = inputs

        # [None, 30] -> [None, 30, 620]
        x = self.embed(x)

        embedding = tf.tanh(x)

        _, forward_h, backward_h = self.bigru(embedding)

        thought_vector = tf.concat([forward_h, backward_h], axis=1)

        return thought_vector, embedding

    def get_config(self):
        config = {'thought_size': self.thought_size,
                  'word_size': self.word_size,
                  'vocab_size': self.vocab_size}
        base_config = super(Encoder, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class DuoDecoder(tf.keras.layers.Layer):

    def __init__(self, max_length, thought_size, word_size, vocab_size, **kwargs):
        super(DuoDecoder, self).__init__(**kwargs)
        self.thought_size = thought_size
        self.word_size = word_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.prev_gru = tf.compat.v1.keras.layers.CuDNNGRU(self.thought_size + self.word_size,
                                                           return_sequences=True,
                                                           return_state=True,
                                                           recurrent_initializer='glorot_uniform')
        self.next_gru = tf.compat.v1.keras.layers.CuDNNGRU(self.thought_size + self.word_size,
                                                           return_sequences=True,
                                                           return_state=True,
                                                           recurrent_initializer='glorot_uniform')
        self.tf_dense1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_size))
        self.tf_dense2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_size))

    def call(self, thoughts, embedding, training=None):
        # [None, thought_size] -> [None, 30, thought_size]
        thoughts = tf.keras.layers.RepeatVector(self.max_length)(thoughts)

        # Prepare Thought Vectors for Prev. and Next Decoders.
        # [None, 30, thought_size] -> [None-1, 30, thought_size]
        prev_thoughts = thoughts[:-1, :, :]
        next_thoughts = thoughts[1:, :, :]

        # Teacher Forcing.
        #   1.) Prepare Word embeddings for Prev and Next Decoders.
        # [None-1, 30, thought_size] -> [None-1, 30, thought_size]
        prev_embeddings = embedding[:-1, :, :]
        next_embeddings = embedding[1:, :, :]

        #   2.) delay the embeddings by one timestep
        # [None-1, 30, 620]
        delayed_prev_embeddings = tf.concat([0 * prev_embeddings[:, -1:, :], prev_embeddings[:, :-1, :]], axis=1)
        delayed_next_embeddings = tf.concat([0 * next_embeddings[:, -1:, :], next_embeddings[:, :-1, :]], axis=1)

        # Supply current "thought" and delayed word embeddings for teacher forcing.
        cat_prev = tf.concat([next_thoughts, delayed_prev_embeddings], axis=2)
        cat_next = tf.concat([prev_thoughts, delayed_next_embeddings], axis=2)

        prev_output, _ = self.prev_gru(cat_prev)
        next_output, _ = self.next_gru(cat_next)

        prev_pred = self.tf_dense1(prev_output)
        next_pred = self.tf_dense2(next_output)

        return prev_pred, next_pred

    def get_config(self):
        config = {'thought_size': self.thought_size,
                  'word_size': self.word_size,
                  'vocab_size': self.vocab_size,
                  'max_length': self.max_length}
        base_config = super(DuoDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class skip_thoughts(tf.keras.Model):

    def __init__(self, thought_size, word_size, vocab_size, max_length, **kwargs):
        super(skip_thoughts, self).__init__(**kwargs)
        self.thought_size = thought_size
        self.word_size = word_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.encoder = Encoder(thought_size, word_size, vocab_size)
        self.decoders = DuoDecoder(max_length, thought_size, word_size, vocab_size)

    def create_mask(self, lengths, max_len, vocab_size):
        mask_list = []
        for i in range(lengths.shape[0]):
            mask_list.append(tf.sequence_mask(lengths[i], max_len))
        mask_list = tf.stack(mask_list)
        mask = tf.cast(tf.tile(tf.expand_dims(mask_list, -1), [1, 1, vocab_size]), dtype=tf.float32)
        return mask

    def call(self, sentences, lengths, training=None):
        # sentences: [None, 30]
        # length: [None]

        # thought_vectors:[None, thought_size, embeddings:[None, 30, word_size]
        thought_vectors, embeddings = self.encoder(sentences)

        # both: [None-1, 30, vocab_size]
        prev_pred, next_pred = self.decoders(thought_vectors, embeddings)

        # mask the predictions, so that loss for beyond-EOS word predictions is cancelled.
        prev_mask = self.create_mask(lengths[:-1], self.max_length, self.vocab_size)
        next_mask = self.create_mask(lengths[1:], self.max_length, self.vocab_size)

        masked_prev_pred = prev_pred * prev_mask
        masked_next_pred = next_pred * next_mask

        return masked_prev_pred, masked_next_pred

