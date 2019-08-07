"""
Configuration file.
"""
import os

thought_size = 1200
embed_dim = 620
vocab_size = 20000
max_length = 30
epochs = 5
# lr = 5e-4
total_sent = 70660978
batch_size_per_gpu = 128/4
validation_size = 0.002
learning_rate = 0.0008,
learning_rate_decay_factor = 0.5,
learning_rate_decay_steps = 400000,
number_of_steps = 500000,
clip_gradient_norm = 5.0

checkpoint_dir = './logs'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")