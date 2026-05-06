# WM811K Wafer Defect Classification

Machine learning project for classifying semiconductor wafer defect patterns using the WM811K wafer map dataset. The project explores dataset loading, wafer map preprocessing, class imbalance, Random Forest classification, and Convolutional Neural Network (CNN) models.

Presented by Meghan Herbert and Najani Johnson.

## Links

- Project repository: https://github.com/sheebos/Machine-Learning-Class-Repo
- Dataset: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
- HTML results report: `WM811K_data_and_analysis/test2_results.html`
- Presentation: `WM811K_data_and_analysis/Wafer Map Defect Classification Presentation.pdf`
- Poster: `WM811K_data_and_analysis/Machine Learning Poster.pdf`

## Background

Semiconductor wafers are tested across many die locations, and those test outcomes can be represented as wafer maps. The spatial layout of failing dies is meaningful: different shapes and regions of failure can point toward different manufacturing problems, such as edge effects, localized process variation, scratches, contamination, or equipment issues.

Wu, Jang, and Chen describe wafer map failure pattern recognition as a way to help engineers identify likely failure causes in large-scale semiconductor manufacturing data. Their work also frames two related analysis goals: recognizing known wafer failure pattern classes and ranking wafer maps by similarity. This project follows the same general motivation by using the WM811K wafer map dataset to classify defect patterns from the wafer map structure.

The dataset is challenging because it is highly imbalanced. Most labeled wafers are marked as `none`, while several defect classes have much smaller sample counts. Because of that imbalance, this project reports not only accuracy but also weighted F1 score, classification reports, confusion matrices, and example predictions.

## Project Contents

```text
.
|-- README.md
|-- ideas.txt
|-- test1.ipynb
|-- LSWMD.pkl
`-- WM811K_data_and_analysis/
    |-- wm811k_metadata.csv
    |-- pickel.py
    |-- pickelopener.ipynb
    |-- pickelopener2.ipynb
    |-- pickelopener_Najanimods.ipynb
    |-- Machine Learning Poster.pdf
    |-- Wafer Map Defect Classification Presentation.pdf
    |-- test2.ipynb
    |-- test2_results.html
    `-- test3.ipynb
```

## File Guide

- `test1.ipynb`: Main exploratory notebook using `LSWMD.pkl`. It loads the wafer dataset, examines class distribution, visualizes wafer maps, removes unlabeled samples, prepares model inputs, and tests baseline and CNN approaches.
- `WM811K_data_and_analysis/test2.ipynb`: Main analysis notebook using a CSV export of the WM811K data. It preprocesses wafer map strings, cleans labels, resizes wafer maps to `26 x 26`, trains/evaluates a Random Forest baseline and CNN models, and creates comparison plots.
- `WM811K_data_and_analysis/test2_results.html`: Standalone HTML report generated from the results in `test2.ipynb`. It includes model metrics, dataset summary, links, presenter information, and embedded plots.
- `WM811K_data_and_analysis/test3.ipynb`: Alternate CNN-focused notebook. It includes environment setup help for the local virtual environment and skips the Random Forest baseline to reduce dependency issues.
- `WM811K_data_and_analysis/Wafer Map Defect Classification Presentation.pdf`: Slide deck covering the project motivation, dataset overview, preprocessing, model implementation, results, overfitting observations, and next steps.
- `WM811K_data_and_analysis/Machine Learning Poster.pdf`: Poster summary of the background, wafer map dataset, Random Forest baseline, CNN model, key takeaways, and references.
- `WM811K_data_and_analysis/pickel.py`: Small script for opening `LSWMD.pkl` with pandas and checking dataset information.
- `WM811K_data_and_analysis/pickelopener*.ipynb`: Utility notebooks for opening and visualizing wafer map data from `.npz` or pickle-derived files.
- `WM811K_data_and_analysis/wm811k_metadata.csv`: Metadata-only CSV containing fields such as die size, lot name, wafer index, train/test label, and failure type.
- `ideas.txt`: Early project notes.

## Dataset

The project uses the WM811K wafer map dataset from Kaggle. Each wafer map represents die-level testing results on a semiconductor wafer. The target labels include defect categories such as:

- `none`
- `Center`
- `Donut`
- `Edge-Loc`
- `Edge-Ring`
- `Loc`
- `Near-full`
- `Random`
- `Scratch`

Large dataset files are intentionally listed in `.gitignore`, including:

- `LSWMD.pkl`
- `WM811K_data_and_analysis/wm811K_csv_export.csv`
- `WM811K_data_and_analysis/wm811k_wafer_maps.npz`
- zipped wafer map exports

To rerun the notebooks from a fresh clone, download the dataset from Kaggle and place the expected files at the paths used by the notebooks.

## Methods

The notebooks follow this general workflow:

1. Load WM811K wafer map data from pickle, CSV, or NPZ files.
2. Clean label columns such as `failureType` and `trianTestLabel`.
3. Remove unlabeled wafer maps for supervised learning.
4. Convert wafer maps into numeric NumPy arrays.
5. Standardize wafer maps to a consistent size, especially `26 x 26`.
6. Encode defect labels using scikit-learn label encoding.
7. Split data into training and testing sets.
8. Train and compare Random Forest and CNN models.
9. Evaluate models using accuracy, weighted F1 score, classification reports, confusion matrices, and example wafer predictions.

## Model Approach

The presentation and poster frame the project around two model types:

- Random Forest baseline: wafer maps are flattened into 1D feature vectors and trained with a 100-tree Random Forest. This gives a strong traditional machine learning comparison point, but it does not directly learn spatial relationships in the wafer image.
- Convolutional Neural Network: wafer maps are treated like small images, allowing the model to learn spatial patterns such as rings, scratches, clusters, and localized failures.

This comparison is important because wafer defects are not only tabular labels; their location and shape across the wafer are part of the signal.

## Results Summary

The generated `test2_results.html` report summarizes the executed `test2.ipynb` run. In that run:

- Processed wafer maps: `74,206`
- Training samples: `59,364`
- Test samples: `14,842`
- Random Forest accuracy: `0.9601`
- Old CNN accuracy: `0.9635`
- Improved CNN accuracy: `0.9299`

The old CNN had the best overall score in the reported run, with `0.9635` accuracy and `0.9631` weighted F1 score.

The presentation and poster also emphasize that training behavior matters beyond final accuracy. The CNN training curves show signs of overfitting: training loss can continue decreasing while validation loss rises. This suggests the model begins memorizing training wafer patterns instead of learning general defect structures.

## Key Takeaways

- Machine learning can help automate wafer defect classification and reduce manual inspection effort.
- Random Forest performs strongly as a baseline, especially for common classes.
- CNNs are a natural fit for wafer maps because they can learn image-like spatial defect patterns.
- Class imbalance is a major challenge because rare defect types have far fewer examples than the `none` class.
- Visually similar patterns can still be confused, so confusion matrices and sample predictions are important for interpretation.

## Next Steps

- Improve CNN generalization with early stopping, dropout, and other regularization methods.
- Address class imbalance using class weighting, oversampling rare classes, or carefully designed augmentation.
- Expand evaluation with more CNN architectures, hyperparameter tuning, and closer review of misclassified defect types.
- Explore similarity ranking ideas from the base research paper to compare visually similar wafer maps.

## Setup

Recommended Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image tensorflow
```

If using Jupyter:

```bash
pip install notebook ipykernel
```

Then open the notebooks in Jupyter or VS Code.

## Notes

Some notebooks contain absolute local paths such as `X:/ECEsite/Machine-Learning-Class-Repo/...`. If running this project on a different machine, update those paths or run notebooks from the repository root and use relative paths.

TensorFlow on native Windows may run on CPU depending on the installed TensorFlow version. The notebook outputs include a warning that TensorFlow GPU support is not available on native Windows for TensorFlow versions `2.11` and newer.

## Reference

Wu, Ming-Ju, Jyh-Shing R. Jang, and Jui-Long Chen. "Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets." *IEEE Transactions on Semiconductor Manufacturing* 28, no. 1 (February 2015): 1-12. https://doi.org/10.1109/TSM.2014.2364237.
