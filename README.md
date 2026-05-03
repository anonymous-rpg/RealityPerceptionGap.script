# RealityPerceptionGap.script

The following script shows how we evaluate the held-out benchmark in the backend.
The input `CSV` file should contain three columns: 1st column: video ID, 2nd column: predicted label (0 for real, 1 for fake), 3rd column: model score
```
python simple_eval.py --pred preds.csv
```
