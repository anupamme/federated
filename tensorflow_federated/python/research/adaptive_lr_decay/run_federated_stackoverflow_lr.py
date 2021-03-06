# Copyright 2020, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trains and evaluates on Stackoverflow LR with adaptive LR decay."""

import functools

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from tensorflow_federated.python.research.adaptive_lr_decay import adaptive_fed_avg
from tensorflow_federated.python.research.adaptive_lr_decay import decay_iterative_process_builder
from tensorflow_federated.python.research.utils import training_loop
from tensorflow_federated.python.research.utils import training_utils
from tensorflow_federated.python.research.utils import utils_impl
from tensorflow_federated.python.research.utils.datasets import stackoverflow_lr_dataset
from tensorflow_federated.python.research.utils.models import stackoverflow_lr_models

with utils_impl.record_hparam_flags():
  # Experiment hyperparameters
  flags.DEFINE_integer('vocab_tokens_size', 10000, 'Vocab tokens size used.')
  flags.DEFINE_integer('vocab_tags_size', 500, 'Vocab tags size used.')
  flags.DEFINE_integer('client_batch_size', 100,
                       'Batch size used on the client.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer(
      'client_epochs_per_round', 1,
      'Number of client (inner optimizer) epochs per federated round.')
  flags.DEFINE_integer(
      'num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')
  flags.DEFINE_integer('max_elements_per_user', 1000, 'Max number of training '
                       'sentences to use per user.')
  flags.DEFINE_integer(
      'max_batches_per_client', -1, 'Maximum number of batches to process at '
      'each client in a given round. If set to -1, we take the full dataset.')
  flags.DEFINE_enum(
      'client_weight', 'uniform', ['num_samples', 'uniform'],
      'Weighting scheme for the client model deltas. Currently, this can '
      'either weight according to the number of samples on a client '
      '(num_samples) or uniformly (uniform).')
  flags.DEFINE_integer(
      'client_datasets_random_seed', 1, 'The random seed '
      'governing the selection of clients that participate in each training '
      'round. The seed is used to generate the starting point for a Lehmer '
      'pseudo-random number generator, the outputs of which are used as seeds '
      'for the client sampling.')

FLAGS = flags.FLAGS


def metrics_builder():
  """Returns a `list` of `tf.keras.metric.Metric` objects."""
  return [
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(top_k=5, name='recall_at_5'),
  ]


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  stackoverflow_train, stackoverflow_validation, stackoverflow_test = stackoverflow_lr_dataset.get_stackoverflow_datasets(
      vocab_tokens_size=FLAGS.vocab_tokens_size,
      vocab_tags_size=FLAGS.vocab_tags_size,
      client_batch_size=FLAGS.client_batch_size,
      client_epochs_per_round=FLAGS.client_epochs_per_round,
      max_training_elements_per_user=FLAGS.max_elements_per_user,
      max_batches_per_user=FLAGS.max_batches_per_client,
      num_validation_examples=FLAGS.num_validation_examples)

  input_spec = stackoverflow_train.create_tf_dataset_for_client(
      stackoverflow_train.client_ids[0]).element_spec

  model_builder = functools.partial(
      stackoverflow_lr_models.create_logistic_model,
      vocab_tokens_size=FLAGS.vocab_tokens_size,
      vocab_tags_size=FLAGS.vocab_tags_size)

  loss_builder = functools.partial(
      tf.keras.losses.BinaryCrossentropy,
      from_logits=False,
      reduction=tf.keras.losses.Reduction.SUM)

  if FLAGS.client_weight == 'uniform':

    def client_weight_fn(local_outputs):
      del local_outputs
      return 1.0

  elif FLAGS.client_weight == 'num_samples':
    client_weight_fn = None
  else:
    raise ValueError('Unsupported client_weight flag [{!s}]. Currently only '
                     '`uniform` and `num_samples` are supported.'.format(
                         FLAGS.client_weight))

  training_process = decay_iterative_process_builder.from_flags(
      input_spec=input_spec,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      client_weight_fn=client_weight_fn)

  client_datasets_fn = training_utils.build_client_datasets_fn(
      stackoverflow_train,
      FLAGS.clients_per_round,
      random_seed=FLAGS.client_datasets_random_seed)

  assign_weights_fn = adaptive_fed_avg.ServerState.assign_weights_to_keras_model

  evaluate_fn = training_utils.build_evaluate_fn(
      model_builder=model_builder,
      eval_dataset=stackoverflow_validation,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      assign_weights_to_keras_model=assign_weights_fn)

  test_fn = training_utils.build_evaluate_fn(
      model_builder=model_builder,
      # Use both val and test for symmetry with other experiments, which
      # evaluate on the entire test set.
      eval_dataset=stackoverflow_validation.concatenate(stackoverflow_test),
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      assign_weights_to_keras_model=assign_weights_fn)

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      training_process, client_datasets_fn, evaluate_fn, test_fn=test_fn)


if __name__ == '__main__':
  app.run(main)
