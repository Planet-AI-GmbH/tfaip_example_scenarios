# ATR Scenario

The ATR-Scenario is an example showing how to implement a line-based ATR-engine.
It provides a CNN/LSTM-network architecture which is trained with the CTC-algorithm.

## Run
To run the training of this scenario execute (in the cloned dir)
```bash
export PYTHONPATH=$PWD  # required so that the scenario is detected
tfaip-train tfaipscenarios.atr
tfaip-train tfaipscenarios.atr --device.gpus 0  # to run training on the first GPU, if available
```

## Data
The [working dir](workingdir) provides some example lines of the UW3 dataset which are loaded by default

## References
* The Open-Source ATR-Engine [Calamari](https://github.com/calamari_ocr/calamari) uses the basic concepts of this example, but is way more sophisticated: Several input sources, voting, dynamic graphs (including dilation, transposed convolution, ...).