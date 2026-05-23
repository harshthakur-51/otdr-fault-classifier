# OTDR Fault Classifier

An edge-friendly demonstration of automated optical fiber fault classification from
OTDR traces. The pipeline generates representative traces, extracts compact signal
features, and trains a lightweight fully connected neural network across eight
event classes.

This repository is a public portfolio implementation inspired by research work on
OTDR fault detection. It uses synthetic traces rather than publication or lab data.

## Fault classes

| Class | Trace signature |
| --- | --- |
| `normal` | Expected attenuation with noise |
| `connector_loss` | Small step loss |
| `splice_loss` | Moderate permanent step loss |
| `bend` | Gradual localized attenuation |
| `reflection` | Sharp reflective pulse |
| `break` | Large loss after a reflective event |
| `dirty_connector` | Repeated local reflections and loss |
| `multiple_events` | Combined loss and reflection events |

## Run

```bash
python -m otdr_fault_classifier.cli --samples-per-class 90 --epochs 90
```

From a fresh checkout, either install the package with `python -m pip install -e .`
or run with `PYTHONPATH=src`.

## Test

```bash
python -m unittest discover -s tests -v
```

## Architecture

- `traces.py`: reproducible OTDR-like signal generation.
- `features.py`: low-compute feature vector extraction.
- `model.py`: standard-library multilayer perceptron and evaluation utilities.
- `cli.py`: train/test execution and confusion matrix output.

The small feature vector and single hidden layer keep the inference path appropriate
for future edge deployment experiments.
