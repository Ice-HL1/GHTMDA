# GHTMDA
A self-supervised heterogeneous graph hierarchical contrastive learning model for efficient metabolite-disease associations prediction

## Requirements:
- Python 3.9.16
- CUDA 11.8.0
- torch 2.0.1
- torch-geometric 2.3.1
- numpy 1.26.1
- pandas 2.2.3
- scikit-learn 1.2.2
- dgl 0.9.1
- networkx 3.2.1
- matplotlib 3.8.0
- seaborn 0.12.2

## Data:
The data files needed to run the model, which contain metabolite-disease association datasets.
- MetaboliteFingerprint, MetaboliteGIP: The similarity measurements of metabolites to construct the similarity network
- DiseasePS, DiseaseGIP: The similarity measurements of diseases to construct the similarity network  
- ProteinSequence, ProteinGIP_Metabolite, ProteinGIP_Disease: The similarity measurements of proteins to construct the similarity network
- MetaboliteDiseaseAssociationNumber: The known metabolite-disease associations
- MetaboliteProteinAssociationNumber: The known metabolite-protein associations
- ProteinDiseaseAssociationNumber: The known protein-disease associations

## HGAE:
- Implementation of Hierarchical Graph AutoEncoder (HGAE)
- Embedding/: The multi-level feature embeddings of metabolites and diseases obtained by HGAE
- Usage: Execute ```python train_HGAE.py``` 

## Code:
- data_preprocess.py: Methods of data processing and heterogeneous graph construction
- metric.py: Evaluation metrics calculation (AUC, AUPR, F1-score, etc.)
- hierarchical_contrast.py: Hierarchical contrastive learning with multi-level sampling
- contrast_learning.py: Self-supervised contrastive learning framework
- graph_augmentation.py: Graph augmentation strategies for contrastive learning
- model.py: GHTMDA model architecture with hierarchical attention mechanism
- train_MDA.py: Training script for metabolite-disease association prediction
- predict.py: Prediction and evaluation script

## Usage:
1. Data preprocessing and graph construction:
   ```python data_preprocess.py```

2. Pre-train hierarchical graph autoencoder:
   ```python train_HGAE.py```

3. Train GHTMDA model:
   ```python train_MDA.py```

4. Make predictions:
   ```python predict.py```

## Model Architecture:
- **Heterogeneous Graph Construction**: Multi-type nodes (metabolites, diseases, proteins) with various edge types
- **Hierarchical Contrastive Learning**: Multi-level graph views with hierarchical positive/negative sampling
- **Self-supervised Framework**: Pre-training with graph reconstruction and contrastive objectives
- **Attention Mechanism**: Cross-attention for metabolite-disease interaction modeling

## Citation:
If you use GHTMDA in your research, please cite:
```
@article{ghtmda2024,
  title={GHTMDA: A self-supervised heterogeneous graph hierarchical contrastive learning model for efficient metabolite-disease associations prediction},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```