#!/bin/sh
pipenv run python training/run_experiment.py --save '{"dataset": "EmnistLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc",
"train_args": {
            "batch_size": 32,
            "epochs": 16
        }
        }' --gpu $1

