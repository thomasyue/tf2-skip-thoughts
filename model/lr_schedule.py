from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=30000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.d_model = tf.cast(self.d_model, tf.float32)

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }
