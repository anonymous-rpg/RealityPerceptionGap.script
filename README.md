# RealityPerceptionGap.script

Ground-truth labels for RPG are not publicly released. Researchers evaluate detectors by submitting predictions to a held evaluation server that returns aggregate metrics. This protects the benchmark's reliability by preventing label leakage into training pipelines and limiting overfitting through repeated evaluation.

The following script shows how we evaluate the hidden-label benchmark in the backend.
The input `CSV` file should contain three columns: 1st column: video ID, 2nd column: predicted label (0 for real, 1 for fake), 3rd column: model score
```
python simple_eval.py --pred preds.csv
```
