# Graph Kernel SVM Experiment Note

This note describes graph classification experiments on MUTAG, PTC, and PROTEINS. The task is
supervised molecular and protein graph classification using support vector machines.

We compare graph statistics features, shortest path kernels, and Weisfeiler-Lehman (WL)
subtree kernels. Models are evaluated with accuracy and macro F1 using repeated stratified
splits.

Results show that WL kernels achieve higher accuracy than graph statistics on MUTAG, while
shortest path kernels remain competitive on PROTEINS. Macro F1 is more variable on PTC because
the class distribution is smaller and noisier.

A reproducibility risk is that kernel hyperparameter grids and exact split seeds are not fully
specified.
