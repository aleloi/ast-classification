import torch
import argparse
import sys
from typing import Tuple, Any

import train
import dataset
import predict
import utils
import linear_lstm_model as linear_model
import dgl_lstm_model as dgl_model

parser = argparse.ArgumentParser(
    description='Train/test codeforces solution classifier.')

parser.add_argument('--mode',
                    required=True,
                    #default='train',
                    help="One out of 'train', 'evaluate', 'predict'. "
                    )


parser.add_argument('--tree', action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Whether to use a tree or a linear model. "
                    "Can't be specified when loading saved model."
                    )

parser.add_argument('--take_top10', action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Whether to use the full 104 classes or small "
                    "10 classes dataset. "
                    "Can't be specified when loading saved model."
                    )

parser.add_argument('--drop_large_threshold_tokens',
                    type=int,
                    help="Whether long programs should be removed from "
                    "the dataset. Default=no programs removed."
                    )

parser.add_argument('--do_prune_duplicates',
                    action=argparse.BooleanOptionalAction,
                    help="Whether to remove duplicates from the "
                    "dataset. ",
                    default=True
                    )

parser.add_argument('--max_per_class',
                    type=int,
                    help="Whether to use a sub-dataset with the first "
                    "`max_per_class` samples of each class. Decreases "
                    "dataset loading time. Default = whole dataset used." 
                    )

parser.add_argument('--training_weight',
                    type=int,
                    default=8,
                    help="Weight for training split. "
                    "Dataset split is deterministic, but depends on "
                    "weights, pruning, drop large tokens arguments. "
                    "Required unless mode is `predict`.",
                    )

parser.add_argument('--validation_weight',
                    type=int,
                    default=2,
                    help="Weight for validation split. "
                    "Required unless mode is `predict`."
                    )

parser.add_argument('--test_weight',
                    type=int,
                    default=1,
                    help="Weight for test split. "
                    "Required unless mode is `predict`."
                    )

parser.add_argument('--train_samples_epoch',
                    type=int,
                    help="Training samples to sample per epoch. "
                    "An epoch is not the whole training set. "
                    "Instead there is weigthed sampling with "
                    "repetition so that each class is equally likely."
                    )

parser.add_argument('--val_samples_epoch',
                    type=int,
                    help="Validation samples to sample per epoch. "
                    "An epoch is not the whole training set. "
                    "Instead there is weigthed sampling with "
                    "repetition so that each class is equally likely."
                    )

parser.add_argument('--test_samples',
                    type=int,
                    help="Test samples to sample. "
                    "An epoch is not the whole training set. "
                    "Instead there is weigthed sampling with "
                    "repetition so that each class is equally likely."
                    )

parser.add_argument('--small_train_samples',
                    type=int,
                    help="Number of samples to generate "
                    "e.g. gradient size tensorboard metrics. "
                    "Is randomly sampled from test split. "
                    "Required when training."
                    )

parser.add_argument('--batch_size',
                    type=int,
                    help="Number of samples to generate "
                    "e.g. gradient size tensorboard metrics. "
                    "Is randomly sampled from test split. "
                    "Required unless mode is `predict`."
                    )

parser.add_argument('--embedding_dim',
                    type=int,
                    help="Embedding dimension that token classes "
                    "are mapped to. Can't be specified when loading "
                    "model. "
                    )

parser.add_argument('--lstm_output_dim',
                    type=int,
                    help="LSTM cell size and output dimension. "
                    "Can't be specified when loading model."
                    )

parser.add_argument('--hidden_fc_layer_dim',
                    type=int,
                    help="Dim of FC layer between LSTM output and "
                    "logits output. Default = no layer. Can't be "
                    "specified when loading model. TODO: model loading "
                    "only works when it's absent or =200. "
                    )

parser.add_argument('--label_smoothing',
                    type=float,
                    default=0.05,
                    help="Label smoothing for the training loss. "
                    "Can't be "
                    "specified when loading model. TODO: model loading "
                    "only works when it's absent or =200. "
                    )

parser.add_argument('--saved_model_dir',
                    type=str,
                    help="A directory, absolute or relative. "
                    "e.g. `results/model_<name>.` "
                    "Can't be "
                    "specified together with model args. "
                    "Loads the latest checkpoint unless --epoch "
                    "is specified. "
                    )

parser.add_argument('--epoch',
                    type=int,
                    help="Model epoch to load. "
                    "Can't be "
                    "specified together with model args. "
                    "If specified, --saved_model_dir must be set. "
                    "Loads the specified checkpoint. "
                    )

parser.add_argument('--predict_path',
                    type=str,
                    help="Path to program that will get class predicted. "
                    "Can only be specified with `--mode predict`. "
                    "Must be specified with `--mode predict`."
                    )

parser.add_argument('--train_epochs',
                    type=int,
                    default=200,
                    help="Number of epochs to train. Ignored unless "
                    "`--mode train`."
                    )

parser.add_argument('--learning_rate',
                    type=float,
                    help="Learning rate. Can only be specified when "
                    "`--mode train`."
                    )

args = parser.parse_args()

def print_and_exit(msg):
    print(msg)
    parser.print_help()
    sys.exit()

def load_model():
    if args.saved_model_dir is None:
        print_and_exit("--saved_model_dir must be specified when "
                       "mode is predict.")
    model = utils.try_build_and_load_model(args.saved_model_dir)
    if args.epoch is not None:
        utils.load_model(model, args.saved_model_dir, epoch)
    return model

def has_all(*arg_list):
    return all(x is not None for x in arg_list)

def has_some(*arg_list):
    return any(x is not None for x in arg_list)

ModelType = Any

def build_or_load_model() -> Tuple[ModelType, str]:
    model: ModelType
    if (has_some(args.saved_model_dir, args.epoch) ^
        has_some(args.embedding_dim, args.lstm_output_dim,
                 args.hidden_fc_layer_dim, args.label_smoothing)):
        print_and_exit("Can't combine loading saved model and "
                       "specifying model architecture")
    if args.saved_model_dir is None:
        if not has_all(args.embedding_dim, args.lstm_output_dim,
                       args.take_top10):
            print_and_exit("Must specify --embedding_dim, "
                           "--lstm_output_dim, --take_top10 "
                           "when creating a new model")                
        # Creating new model.
        model_args = {'embedding_dim': args.embedding_dim,
                      'lstm_output_dim': args.lstm_output_dim,
                      'num_classes': 10 if args.take_top10 else 104,

                      # These are None when not specified, which is what
                      # the model classes expect.
                      'hidden_fc_layer_dim': args.hidden_fc_layer_dim,
                      'label_smoothing': args.label_smoothing}
        if args.tree:
            model = dgl_model.DGLTreeLSTM(**model_args)
        else:
            model = linear_model.LinearLSTM(**model_args)
        results_dir = utils.create_model_directory(
            get_ds_args(),
            args.learning_rate,
            model
            )
    else:
        model, results_dir = utils.try_build_and_load_model(
            args.saved_model_dir)
        if args.epoch is not None:
            results_dir = utils.load_model(model,
                                           args.saved_model_dir, args.epoch)
    return model, results_dir

def get_ds_args() -> dataset.DataArgs:
    # All other required args have default values.
    if not has_all(args.batch_size,
                   args.train_samples_epoch,
                   args.val_samples_epoch,
                   args.test_samples,
                   args.small_train_samples
                   ):
        print_and_exit(f"--batch_size, "
                       "--(train|val|test)_ samples[_epoch] and "
                       "--small_train_samples "
                       "required when loading dataset.")
    return dataset.DataArgs(
        flatten=not args.tree,
        take_top10=args.take_top10,
        training_weight = args.training_weight,
        validation_weight = args.validation_weight,
        test_weight = args.test_weight,
        train_samples = args.train_samples_epoch,
        val_samples = args.val_samples_epoch,
        test_samples = args.test_samples,
        small_train_samples = args.small_train_samples,
        batch_size=args.batch_size,
        drop_large_threshold_tokens=args.drop_large_threshold_tokens,
        do_prune_duplicates = args.do_prune_duplicates,
        max_per_class=args.max_per_class
    )

if args.mode not in ['train', 'predict', 'evaluate']:
    print_and_exit(f"--mode is {args.mode} not one of "
                   "'train', 'predict', 'evaluate'")

if args.mode == 'predict':
    if args.predict_path is None:
        print_and_exit("--predict_path must be specified "
                       "when mode is predict")
    model, _ = load_model()
    predict.predict(model, args.predict_path)
    
elif args.mode == 'train':
    if args.learning_rate is None:
        print_and_exit("--learning_rate required when training")
    if args.predict_path is not None:
        print_and_exit("Can't specify --predict_path when training")
    model, results_dir = build_or_load_model()
    print(f"RESULTS_DIR is {results_dir}")
    print(model)
    dl_train, dl_train_small, dl_val, dl_test = dataset.get_datasets(
        get_ds_args())
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate  # type: ignore
                                 )
    train.train(model,
                0 if args.epoch is None else args.epoch,
                args.train_epochs,
                optimizer=optimizer,
                dl_train=dl_train,
                dl_train_small=dl_train_small,
                dl_val=dl_val,
                results_dir=results_dir)
else:
    print_and_exit(f"--mode={args.mode} is not implemented")
