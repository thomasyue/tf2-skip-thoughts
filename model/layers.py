from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class GRUwithLayerNorm(tf.keras.layers.GRUCell):
    """GRU cell with layer normalization.
    Layer normalization implementation based on:
    https://arxiv.org/abs/1607.06450.
    "Layer Normalization"
    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
    """

    def __init__(self, units, **kwargs):
        super(GRUwithLayerNorm, self).__init__(units, **kwargs)
        self.units = units
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.layernorm3 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, states, training):
        from tensorflow.python.ops import array_ops
        from tensorflow.python.keras import backend as K

#         inputs, states = super().call(inputs, states)

        h_tm1 = self.layernorm(states[0])  # previous memory

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=3)

        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = array_ops.unstack(self.bias)

        if self.implementation == 1:
            if 0. < self.dropout < 1.:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs

            x_z = K.dot(inputs_z, self.kernel[:, :self.units])
            x_r = K.dot(inputs_r, self.kernel[:, self.units:self.units * 2])
            x_h = K.dot(inputs_h, self.kernel[:, self.units * 2:])

            if self.use_bias:
                x_z = K.bias_add(x_z, input_bias[:self.units])
                x_r = K.bias_add(x_r, input_bias[self.units: self.units * 2])
                x_h = K.bias_add(x_h, input_bias[self.units * 2:])

            if 0. < self.recurrent_dropout < 1.:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1

            recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel[:, :self.units])
            recurrent_r = K.dot(h_tm1_r,
                                self.recurrent_kernel[:, self.units:self.units * 2])
            if self.reset_after and self.use_bias:
                recurrent_z = K.bias_add(recurrent_z, recurrent_bias[:self.units])
                recurrent_r = K.bias_add(recurrent_r,
                                         recurrent_bias[self.units:self.units * 2])

            z = self.recurrent_activation(self.layernorm1(x_z + recurrent_z))
            r = self.recurrent_activation(self.layernorm2(x_r + recurrent_r))
#             z = self.recurrent_activation(x_z + recurrent_z)
#             r = self.recurrent_activation(x_r + recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel[:, self.units * 2:])
                if self.use_bias:
                    recurrent_h = K.bias_add(recurrent_h, recurrent_bias[self.units * 2:])
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = K.dot(r * h_tm1_h,
                                    self.recurrent_kernel[:, self.units * 2:])

            hh = self.activation(self.layernorm3(x_h + recurrent_h))
#             hh = self.activation(x_h + recurrent_h)
        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]

            # inputs projected by all gate matrices at once
            matrix_x = K.dot(inputs, self.kernel)
            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                matrix_x = K.bias_add(matrix_x, input_bias)

            x_z = matrix_x[:, :self.units]
            x_r = matrix_x[:, self.units: 2 * self.units]
            x_h = matrix_x[:, 2 * self.units:]

            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner = K.bias_add(matrix_inner, recurrent_bias)
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel[:, :2 * self.units])

            recurrent_z = matrix_inner[:, :self.units]
            recurrent_r = matrix_inner[:, self.units:2 * self.units]

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            if self.reset_after:
                recurrent_h = r * matrix_inner[:, 2 * self.units:]
            else:
                recurrent_h = K.dot(r * h_tm1,
                                    self.recurrent_kernel[:, 2 * self.units:])

#             hh = self.activation(x_h + recurrent_h)
            hh = self.activation(self.layernorm3(x_h + recurrent_h))
        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]


class Encoder(tf.keras.layers.Layer):

    def __init__(self, thought_size, word_size, vocab_size, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.thought_size = thought_size
        self.word_size = word_size
        self.vocab_size = vocab_size
        self.embed = tf.keras.layers.Embedding(self.vocab_size, self.word_size)
        # self.bigru = tf.keras.layers.Bidirectional(
        #     tf.compat.v1.keras.layers.CuDNNGRU(self.thought_size, return_sequences=False, return_state=True)
        # )
        self.bigru = tf.keras.layers.Bidirectional(
            tf.keras.layers.RNN(GRUwithLayerNorm(self.thought_size), return_sequences=False, return_state=True)
        )

    def call(self, sentences, training=None):
        # sentence = [None, 30] with zero post padded
        x = sentences

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
        # self.thought_size = thought_size
        self.thought_size = thought_size * 2
        self.word_size = word_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        # self.prev_gru = tf.compat.v1.keras.layers.CuDNNGRU(self.thought_size,
        #                                                    return_sequences=True,
        #                                                    return_state=True,
        #                                                    recurrent_initializer='glorot_uniform')
        # self.next_gru = tf.compat.v1.keras.layers.CuDNNGRU(self.thought_size,
        #                                                    return_sequences=True,
        #                                                    return_state=True,
        #                                                    recurrent_initializer='glorot_uniform')
        self.prev_gru = tf.keras.layers.RNN(GRUwithLayerNorm(self.thought_size),
                                           return_sequences=True,
                                           return_state=True)
        self.next_gru = tf.keras.layers.RNN(GRUwithLayerNorm(self.thought_size),
                                           return_sequences=True,
                                           return_state=True)
        self.td_dense1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_size))
        self.td_dense2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_size))

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

        prev_pred = self.td_dense1(prev_output)
        next_pred = self.td_dense2(next_output)

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

