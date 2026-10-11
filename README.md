# cortex-decoder: Decoding Visual Object Categories from fMRI

Can we tell what a person was looking at from the pattern of activity in their ventral temporal cortex? This project decodes 8 visual object categories from fMRI using linear classifiers, with leave-one-run-out cross-validation, a baseline, a permutation test, and voxel weight maps.

73.3% ± 8.2% accuracy on 8-way object classification (chance = 12.5%) using L2-regularized logistic regression, evaluated on held-out scanning runs (leave-one-run-out CV, single subject).

#### Background

##### Haxby et al. (2001) 
showed that different object categories evoke distinct, distributed and overlapping patterns of activity across ventral temporal cortex, and that category can be read out from these patterns even without the most category-selective voxels. This project is a small decoding replication of that idea on the public dataset from the paper, using modern tooling.

#### Data
Dataset: Haxby et al. (2001), subject 1, loaded with nilearn.datasets.fetch_haxby().
Design: 12 runs. In each run the subject viewed 8 categories (face, cat, house, chair, scissors, shoe, bottle, scrambled pictures), separated by rest periods.
Region: ventral temporal cortex mask supplied with the dataset (464 voxels).
Samples: 864 timepoints after removing rest (108 per category, balanced).

#### Methods
Masking and preprocessing: NiftiMasker with the ventral temporal mask, detrending, and z-scoring per voxel. This is applied to the full time series, and rest rows are removed afterwards so detrending sees an unbroken signal.
Rest removed: rest is the largest class (588 of 1452 timepoints). Keeping it would create an imbalanced 9-class problem with a ~40% majority baseline and inflate accuracy. With 8 balanced classes, chance is exactly 12.5%.
Validation: leave-one-run-out cross-validation (LeaveOneGroupOut, 12 folds). Neighboring fMRI timepoints are correlated, so random splits would leak information.
Models: multinomial logistic regression (L2, C=1, max_iter=1000) and linear SVC for comparison, plus a majority-class DummyClassifier.
Significance: permutation_test_score with the same cross-validation scheme.
Interpretation: a model fit on all data gives one weight map per category, projected back to brain space with masker.inverse_transform.

#### Results
Sensitivity to regularization

Accuracy rises steeply as regularization weakens and then plateaus, so the headline number does not depend on a lucky C.

C	Accuracy (mean ± std)
0.0001	31.8% ± 10.2%
0.001	48.1% ± 9.0%
0.01	65.7% ± 9.2%
0.1	73.6% ± 8.2%
1 (used for the headline result)	73.3% ± 8.2%
10	71.6% ± 8.1%

C=1 was fixed before running the sweep. The difference between C=0.1 and C=1 is far smaller than the run-to-run spread.

Per-category performance
Category	Precision	Recall	F1
scrambled pictures	0.97	0.90	0.93
house	0.92	0.92	0.92
face	0.84	0.80	0.82
cat	0.79	0.77	0.78
shoe	0.65	0.67	0.66
chair	0.71	0.60	0.65
scissors	0.62	0.56	0.59
bottle	0.47	0.66	0.55

Scrambled pictures, houses, faces and cats are decoded most reliably. Errors concentrate among the small man-made objects (shoe, chair, scissors, bottle), consistent with the overlapping category representations described by Haxby et al. Bottle has low precision, meaning many other objects are labeled as bottle. Differences of a few F1 points between categories are within noise, so the tiers are more reliable than the exact ranking.

##### Figures
File	What it shows
lr_accuracy_per_run.png	Accuracy on each held-out run, logistic regression vs SVM vs chance
lr_confusion_matrix.png	Confusion matrix, counts and row-normalized
lr_f1_per_class.png	Per-category F1
lr_accuracy_vs_C.png	Accuracy vs regularization strength
lr_permutation_test.png	Real score against the permutation null
lr_weights_grid.png	Weight map per category on a common slice
lr_weights_ortho_*.png, lr_weights_mosaic_face.png	Orthogonal and mosaic views
lr_weights_preferred_category.png	Voxels colored by the category with the largest weight
lr_weights_heatmap.png, lr_weights_correlation.png	Weight heatmap and similarity between category maps

#### Limitations
One subject. The results may not generalize across people. Extending to more subjects (fetch_haxby(subjects=[1, 2, 3])) is the most valuable next step.
Decoding shows information is present, not how the brain uses it.
Weight maps show where the decoder looks. They come from a model fit on all data and are not a precise map of category selectivity. Logistic regression is multinomial here, so each map reflects a category's weight relative to the others.
Model comparison is not fully controlled. The SVM used its default C and was not tuned, so the gap to logistic regression may partly reflect tuning, not the algorithm.
Hyperparameter sweep used the same cross-validation as the reported score, which is why the headline C was fixed in advance.
Subject-space coordinates. Data are in the subject's native space, so slice positions are not MNI coordinates.
Permutation p-value is limited by the number of permutations.

#### Reproducing
bash
pip install nilearn scikit-learn matplotlib pandas numpy
jupyter notebook haxby_fmri_decoding_logreg.ipynb

The dataset downloads automatically on first run. Settings (C, number of permutations, number of parallel jobs) are at the top of the notebook. The permutation test is the slowest step, so reduce N_PERMUTATIONS for a quick first run.

#### Repository contents
Final.ipynb   # main notebook (Logistic Regression, SVM comparison, all plots)
LogisticRegression.ipynb # Exploratory Logistic Regression model for faces vs houses
SVC.ipynb # Exploratory SVC model for faces vs houses
README.md
*.png                              # figures saved by the notebooks
requirements.txt # Required libaries to be installed

##### Possible extensions
Repeat for multiple subjects and report per-subject accuracy.
Average timepoints within each stimulus block to improve signal-to-noise.
Compare ventral temporal against a whole-brain mask.
Reproduce the paper's exclusion analysis: remove the most category-selective voxels and re-decode.
Tune C for both models with nested cross-validation for a fairer comparison.

#### Reference

Haxby, J. V., Gobbini, M. I., Furey, M. L., Ishai, A., Schouten, J. L., & Pietrini, P. (2001). Distributed and overlapping representations of faces and objects in ventral temporal cortex. Science, 293(5539), 2425-2430.
