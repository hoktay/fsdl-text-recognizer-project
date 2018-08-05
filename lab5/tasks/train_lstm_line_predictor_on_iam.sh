#!/bin/sh
pipenv run python training/run_experiment.py --save --gpu -1 '{"dataset": "IamLinesDataset", 
"model": "LineModelCtc", 
"network": "line_lstm_ctc",
"network_args": {
"window_width": 6,
"window_stride": 2
},
"train_args": {
            "batch_size": 64,
            "epochs": 64,
            "user_name": "hoktay"
        }
        }'