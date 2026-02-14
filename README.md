# MOCHEE: Accurate Source-Free Speech Classification via Meta-Learned Target-Centric Model Merging

This repository contains the implementation of MOCHEE, a target-centric
model merging framework for source-free speech classification.

------------------------------------------------------------------------

## Environment

Recommended setup:

-   python==3.9
-   torch\>=2.0
-   torchaudio
-   transformers
-   numpy
-   scikit-learn
-   scipy
-   pandas
-   tqdm


------------------------------------------------------------------------

## Code Structure

-   main.py : Entry script for model merging and evaluation.

-   src/data.py : Loads pretrained source classifiers and target feature caches.

-   src/merge.py: Implements permutation alignment and meta-learned source weighting.

-   src/train.py: Target-domain evaluation and tuning logic.

-   src/utils.py: Utility functions (seed control, parameter handling).


------------------------------------------------------------------------

## Running the Code

Example:

python run_merge.py\
--target <target_dataset>\
--sources <source1> <source2> ...\
--seeds 1 4 7\
--tgt_val_ratio 0.05\
--tgt_test_ratio 0.9\
--device cuda:0

Main arguments:

-   --target: target dataset name
-   --sources: list of pretrained source classifiers
-   --seeds: random seeds
-   --lr: learning rate

------------------------------------------------------------------------

## Notes

-   This implementation assumes source-domain data are unavailable.
-   Only pretrained source classifiers and limited labeled target data are used.
-   Evaluation reports accuracy and macro-F1.

------------------------------------------------------------------------

## Citation

Anonymous submission.
