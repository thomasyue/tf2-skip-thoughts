import os
import pickle
import datetime
import warnings
from config import *
from tqdm import tqdm
import tensorflow as tf
from model import layers
from model import lr_schedule
from data_loader import DataLoader
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

# from absl import flags
# FLAGS = flags.FLAGS


def train():
    def compute_loss(labels, predictions):
        loss_fn = tf.keras.losses.sparse_categorical_crossentropy
        per_example_loss = loss_fn(labels, predictions, from_logits=True)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

    def train_step(inputs, optimizer):

        sentences, lengths = inputs

        with tf.GradientTape() as tape:
            masked_prev_pred, masked_next_pred = model(sentences, lengths)

            prev_loss = compute_loss(sentences[:-1, :], masked_prev_pred)
            next_loss = compute_loss(sentences[1:, :], masked_next_pred)
            losses = prev_loss + next_loss

        grads = tape.gradient(losses, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return losses

    def val_step(inputs):
        sentences, lengths = inputs
        masked_prev_pred, masked_next_pred = model(sentences, lengths)

        prev_loss = compute_loss(sentences[:-1, :], masked_prev_pred)
        next_loss = compute_loss(sentences[1:, :], masked_next_pred)
        losses = prev_loss + next_loss

        return losses

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs, optimizer))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_test_step(dataset_inputs):
        return strategy.experimental_run_v2(val_step, args=(dataset_inputs,))

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        total_test_loss = 0.0
        num_test_batches = 0
        total_training_steps = (total_sent - val_size) // global_batch_size
        total_testing_steps = val_size // global_batch_size
        for step, x in tqdm(enumerate(train_dist_dataset), total=total_training_steps):

            total_loss += distributed_train_step(x)
            num_batches += 1
            batch_loss = total_loss / num_batches
            if step % 100 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', batch_loss, step=step)
                template = ("Epoch {}, Step {}, Batch Loss: {}")

                print(template.format(epoch + 1, step, batch_loss))
            if step % 15000 == 0:
                checkpoint.save(checkpoint_prefix)

        for step, x in tqdm(enumerate(val_dist_dataset), total=total_testing_steps):
            total_test_loss += distributed_test_step(x)
            num_test_batches += 1
            test_batch_loss = total_test_loss/num_test_batches

            if step % 100 == 0:
                with test_summary_writer.as_default():
                    tf.summary.scalar('val_loss', test_batch_loss, step=step)
                template = ("Epoch {}, Step {}, Val Loss: {}")

                print(template.format(epoch + 1, step, test_batch_loss))



        train_loss = total_loss / num_batches
        if epoch % 1 == 0:
            checkpoint.save(shcekpoint_prefix)
            template = ("Epoch {}, Total Loss: {}")
            print(template.format(epoch + 1, train_loss))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        metavar="<command>",
                        help="train from scratch or continue from last ckpt")
    parser.add_argument('--gpu',
                        required=True,
                        type=str,
                        metavar="choose which gpu to train on")

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Visible GPUs: ", args.gpu)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Create Distributed training strategy
    strategy = tf.distribute.MirroredStrategy()
    print('Number of GPUs: {}'.format(strategy.num_replicas_in_sync))
    global_batch_size = batch_size_per_gpu * strategy.num_replicas_in_sync
    print("Global batch size: {}".format(global_batch_size))

    # Load or create model
    with strategy.scope():
        model = layers.skip_thoughts(thought_size=thought_size, word_size=embed_dim, vocab_size=vocab_size,
                                     max_length=max_length)

        lr = lr_schedule.CustomSchedule(16)
        optimizer = tf.optimizers.Adam(learning_rate=lr)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        if args.command == 'continue':
            ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
            print("Loading model from ", ckpt_path)
            checkpoint.restore(ckpt_path)
        elif args.command == 'train':
            print("Training model from scratch")
            pass
        else:
            print("Please enter 'train' or 'continue'!")
            print("Now Training model From scratch")
            pass

        try:
            model.summary()
        except:
            pass

    # load data, and train test split them.
    print("Loading text files")
    with open('./data/total_sentences', 'rb') as t:
        total_sentences = pickle.load(t)
    with open('./data/lengths', 'rb') as l:
        lengths = pickle.load(l)

    print("Creating Tensorflow 2.0 datasets & distributed training strategy")
    dataset = tf.data.Dataset.from_tensor_slices((total_sentences, lengths))
    val_size = int(validation_size * len(total_sentences))
    print('Validation size: {}'.format(val_size))

    if val_size > 0:
        val_dataset = dataset.take(val_size).batch(global_batch_size)
        train_dataset = dataset.skip(val_size).batch(global_batch_size)

        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
        val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)
    else:
        dataset = dataset.batch(global_batch_size)
        train_dist_dataset = strategy.experimental_distribute_dataset(dataset)

    current_time = datetime.datetime.now().strftime("%m%d-%H%M")
    train_log_dir = './logs/gradient_tape/' + current_time + '/train'
    test_log_dir = './logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Start training
    with strategy.scope():
        train()

