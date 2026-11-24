# Experiments (not part of the final pipeline)

Everything we tried before settling on the final models in `task1/` … `task5/`, kept for the record and for the
technical report. Nothing here is needed to rebuild the final submission.

```
experiments/
├── taskN/experiments.md   dated log of every experiment: change, CV score, LB score if uploaded, keep / drop
├── taskN/src/             experiment and analysis scripts (rounds r1, r2, …)
├── taskN/outputs/         alternative prediction files and CV tables from those experiments
├── task4/RECIPE.md        the task-4 training recipe as it evolved (v47 → v61)
├── task4/notebooks/       earlier task-4 notebooks: CPU baseline (frozen ConvNeXt-T + logistic regression),
│                          round-1 ConvNeXt-T fine-tune, and the first final notebook (blend via a script)
├── task4/src/             scene-held-out CV, benchmarks, blend scripts, cloud-GPU job kits, queue / launch scripts
├── task4/outputs/         training logs and results of every task-4 run that is not in the final blend
└── scripts/inspect_ids.py check of the test-id → submission-row mapping
```

The scripts are archived as they were run, from `taskN/src/` in the old layout: they locate the repository with
`Path(__file__).parents[2]` and import their neighbours from the same folder. To rerun one, copy it back into
`taskN/src/` first. The Kaggle upload history with every public LB score is in `../submissions/log.md`.
