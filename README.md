# Enhanced detection and classification of Human Values Violation in App Reviews

This repository contains datasets and code for identifying and classifying human values violations in app reviews. The project focuses on developing DL, Transformer models to detect these violations and then their fusion by different ensemble techniques to categorize the value violations effectively.

## Directory Structure

### Data Files
- **amazonbinaryclassificationdataset_DL.csv**: Dataset for DL models binary classification of human values violations.
- **amazonbinaryclassificationdataset_transformer.csv**: Dataset for Transformer models binary classification of human values violations.
- **newextendeddataset.csv**: Extended dataset for multiclass classification amd Ensemble techniquestasks.

### Documentation
- **Coding guideline for human values violation Dataset Annotation.pdf**: Detailed guidelines for annotating datasets for human values violation detection.

### Code Files

#### Binary_Experiment
- **`BIGRU_Binary.ipynb`**: Implements Bidirectional GRU for binary classification.
- **`BiLSTM_Binary.ipynb`**: Uses BiLSTM architecture for binary classification tasks.
- **`CNN_Binary.ipynb`**: Employs a Convolutional Neural Network for binary classification.
- **`GRU_Binary.ipynb`**: GRU-based model for binary classification.
- **`LSTM_Binary.ipynb`**: LSTM model implementation for binary classification.
- **`Bert_Binary.ipynb`**: Transformer-based model for binary classification.
- **`Distlbert_Binary.ipynb`**: Transformer-based model for binary classification.

#### MultiClassification_Experiment
- **`GRU, BiGRU, LSTM, BiLSTM, CNN_multiclassification_for humanValues.ipynb`**: Implements GRU, BiGRU, and CNN for multiclass classification of human values violations.
- **`BERT&DistilBERT__MultiClassification_for humanValues.ipynb`**: Transformer-based BERT and Distilbert models for multiclass classification tasks.
  
#### Ensemblew Techniques
- **`Soft and hard Voting Fusion for humanValues.ipynb`**: Implements soft and hard voting fusion of Heterogenous models (BiGRU, CNN, BERT and Distilbert for improving test accuray of multiclass classification for human values violations.

## Requirements
- Os: macOS (M1/M2/M3), Linux, or Windows (WSL2 recommended for GPU)
- Python: >=3.9, <3.12 (3.10/3.11 ideal)
- Jupyter Notebook or JupyterLab
- Core packages accross models:`pandas, numpy, scikit-learn, matplotlib, seaborn, nltk, imbalanced-learn, tqdm
- # DL Models Specific Dependencies:
- Framework: TensorFlow/Keras
- Key Libraries: tensorflow>=2.12, tensorflow-metal==1.1.* (macOS only)
- Installation Notes: On Apple Silicon, install tensorflow-macos + tensorflow-metal for GPU acceleration.
# Transformer Models Specific Dependencies:
- Framework: Hugging Face Transformers (PyTorch)
- Key Libraries: torch>=2.0, transformers>=4.30, accelerate, beautifulsoup4, contractions.
- Installation Notes: MPS enabled via torch.device("mps"); no CUDA required on macOS, Uses custom BERTClassifier(nn.Module) (not
  BertForSequenceClassification)

## Running the Experiments
amazonbinaryclassificationdataset_DL.csv**: Dataset for DL models binary classification of human values violations.
- **amazonbinaryclassificationdataset_transformer.csv**: Dataset for Transformer models binary classification of human values violations.
### Binary Classification
1. Use **amazonbinaryclassificationdataset_DL.csv** For DL & **amazonbinaryclassificationdataset_transformer.csv** for transformer models as the input dataset.(Dataset are same just saved on different names)
2. Choose a model notebook from the **Binary_Experiment models**:
   - Examples: **`GRU_Binary.ipynb`**, **`CNN_Binary.ipynb`**
3. Follow the instructions in the notebook to preprocess data, train the model, and evaluate results.

### Multiclass Classification
1. Use **newextendeddataset.csv** for multiclass classification experiments.
2. Select a notebook from the **Multiclassification Experiment Models**:
   - DL Examples: **`GRU, BiGRU, LSTM, BiLSTm and CNN_Multiclassification.ipynb`**
   - TL Examples: **`BERT&DistilBert_Multiclassification.ipynb`**
3. Execute the notebook to explore multiclass classification approaches and evaluate model performance.

### Ensemble Techniques Classification
1. Use same **newextendeddataset.csv** for multiclass classificatio.
2. Select a notebook from the **Ensemble Models**:
   - For Soft Voting Fusion: **`Ensemble_1 _SoftVoting(Average Probabilities).ipynb`**
   - For Hard Votin Fuison: **`Ensemble_1 _HardVoting(Majority Vote).ipynb`**
   - Load the save models (final_cnn_model.h5, final_cnn_tokenizer.pkl, final_bigru_model.h5, final_bigru_tokenizer.pkl, bert_best.zip and distilbert_best_grid.zip).
3. Execute the notebook to explore enhanced ensemble multiclass classification approaches and evaluate model performance.

## Data Annotation
For guidelines on annotating datasets for human values violations, refer to **Coding guideline for human values violation Dataset Annotation.pdf**.
