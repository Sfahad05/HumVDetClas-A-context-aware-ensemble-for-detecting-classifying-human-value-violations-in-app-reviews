# HumVDetClas: A Context-Aware Heterogeneous Ensemble for Detecting and Classifying Human Value Violations in App Reviews

This repository contains the code and datasets for our published work:

**Khan, S. F., Wang, L., Khan, J. A., Iqbal, A., Khan, N. D., & Zhang, D. (2026). _HumVDetClas: A Context-Aware Heterogeneous Ensemble for Detecting and Classifying Human Value Violations in App Reviews_. Expert Systems with Applications, 133108.**

---

## Repository Overview

The repository is organized into three main experiment stages:

1. **Binary violation detection** (Violation vs. Non-violation)
2. **Multiclass value-violation classification**
3. **Ensemble fusion** (hard voting and soft voting) for improved multiclass performance

### Top-level structure

- `Binary_experiment/`  
  Binary classification experiments and related datasets.
- `Multiclassification_experiment/`  
  Multiclass classification experiments and extended dataset.
- `Ensemble_techniques/`  
  Final ensemble notebooks (hard and soft voting).
- `Coding guidline for human values violation Dataset Annotation.pdf`  
  Annotation guideline used for human value violation labeling.

---

## Datasets

### Binary task datasets (`Binary_experiment/`)
- `amazonbinaryclassificationdataset_DL.csv`  
  Binary dataset prepared for deep learning models.
- `amazonbinaryclassificationdataset_transformer.csv`  
  Binary dataset prepared for transformer models.

### Multiclass task dataset (`Multiclassification_experiment/`)
- `AmazonHVV.csv: A Manually Annotated Benchmark Dataset of Human Value Violations in Low-Rated Amazon App Reviews`  
  An expert-annotated benchmark for ten-class classification of human value violations in low-rated Amazon App Store reviews..

---

## Experiment Flow (How to Reproduce)

To replicate the full HumVDetClas pipeline, run experiments in this order:

### Step 1: Binary detection experiments
Folder: `Binary_experiment/`

Notebooks:
- `K-FoldCNN_Binary.ipynb`
- `K-FoldLSTM_Binary.ipynb`
- `K-FoldBiLSTM_Binary.ipynb`
- `k-FoldGRU_Binary.ipynb`
- `k-FoldBigru_Binary.ipynb`
- `K-FoldBert_Binary.ipynb`
- `K-FoldDistilBert_Binary-2.ipynb`

Purpose:
- Train/evaluate baseline deep learning and transformer models for binary violation detection.
- Produce comparative performance metrics for model selection.

### Step 2: Multiclass classification experiments
Folder: `Multiclassification_experiment/`

Notebooks:
- `K-foldCNN_Multiclassification.ipynb`
- `K-FoldLSTM_Multiclassification.ipynb`
- `K-FoldBILSTM_Multiclassification.ipynb`
- `K-FoldGRU_Multiclassification.ipynb`
- `KFold_GridBiGRU_Multiclass.ipynb`
- `K-FoldBERT_Multiclassification.ipynb`
- `K-FoldGridDistilBert_Multiclassification.ipynb`

Purpose:
- Train/evaluate individual heterogeneous base learners for multiclass value categories.
- Generate model-level outputs for downstream ensemble fusion.

### Step 3: Ensemble fusion experiments
Folder: `Ensemble_techniques/`

Notebooks:
- `Final_Ensemble_2 _HardVoting(Majority Vote).ipynb`
- `Final_soft_voting_fusion_with_ablation_table.ipynb`

Purpose:
- Combine predictions from heterogeneous models.
- Evaluate hard-voting and soft-voting variants.
- Reproduce final ensemble outcomes and ablation-style comparisons.

---

## Suggested Execution Order

For best reproducibility, execute notebooks in the following order:

1. Binary notebooks (`Binary_experiment/`)  
2. Multiclass notebooks (`Multiclassification_experiment/`)  
3. Ensemble notebooks (`Ensemble_techniques/`)  

> Note: Some ensemble steps assume prediction outputs or saved artifacts from base multiclass models. Keep a consistent experiment directory structure and random seeds while reproducing results.

---

## Environment Requirements

- Python 3.9+ (recommended 3.10/3.11)
- Jupyter Notebook / JupyterLab
- Core packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`
- Deep learning stack: `tensorflow` / `keras`
- Transformer stack: `torch`, `transformers`, `accelerate`

Install dependencies according to the notebook imports in each folder (binary, multiclass, and ensemble) before running experiments.

---

## Citation Policy (Important)

If you use this repository to:
- replicate the experiments,
- reuse any part of the code,
- use, modify, extend, or redistribute the dataset(s),

**you must cite our paper**.

### BibTeX
```bibtex
@article{khan2026humvdetclas,
  title={HumVDetClas: A Context-Aware Heterogeneous Ensemble for Detecting and Classifying Human Value Violations in App Reviews},
  author={Khan, Shah Fahad and Wang, Lei and Khan, Javed Ali and Iqbal, Anjum and Khan, Nek Dil and Zhang, Dongyu},
  journal={Expert Systems with Applications},
  pages={133108},
  year={2026},
  publisher={Elsevier}
}
```

---

## Notes

- Please consult `Coding guidline for human values violation Dataset Annotation.pdf` for annotation standards and label interpretation.
- If you publish derivative work based on this repository, include both citation and a brief statement of what was changed (e.g., preprocessing, architecture, or class schema).
