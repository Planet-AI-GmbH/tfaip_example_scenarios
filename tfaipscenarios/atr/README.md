# ATR Scenario

The ATR-Scenario is an example showing how to implement a line-based ATR-engine.
It provides a CNN/LSTM-network architecture which is trained with the CTC-algorithm.

## Run
To run the training of this scenario execute (in the cloned dir)
```bash
export PYTHONPATH=$PWD  # required so that the scenario is detected

# Training
tfaip-train tfaipscenarios.atr --trainer.output_dir atr_model
tfaip-train tfaipscenarios.atr --trainer.output_dir atr_model --device.gpus 0  # to run training on the first GPU, if available

# Validation (of the best model)
tfaip-lav --export_dir atr_model/best --data.image_files tfaipscenarios/atr/workingdir/uw3_50lines/test/*.png

# Prediction
tfaip-predict --export_dir atr_model/best --data.image_files tfaipscenarios/atr/workingdir/uw3_50lines/test/*.png
```

Note, the prediction will only print the raw output of the network.

## Data
The [working dir](workingdir) provides some example lines of the UW3 dataset which are loaded by default

## References
* The Open-Source ATR-Engine [Calamari](https://github.com/calamari_ocr/calamari) uses the basic concepts of this example, but is way more sophisticated: Several input sources, voting, dynamic graphs (including dilation, transposed convolution, ...).