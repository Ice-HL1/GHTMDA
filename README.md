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
adj: Known metabolite-disease associations matrix
similarityies of disease:
disease _information_entropy_similarity ：Disease similarity based on information entropy of associated metabolites
disease_GIP_similarity： Gaussian Interaction Profile (GIP) similarity for diseases
disease_semantic_similarity：Semantic similarity based on disease ontology and medical knowledge
similarityies of meatabolite:
metabolite_entropy_sim: Entropy-based similarity considering metabolite interaction patterns
metabolite_gip_sim：Chemical structure similarity based on molecular descriptors and fingerprints
metabolite_structure_sim: Chemical structure similarity based on molecular descriptors and fingerprints

## Code:
- data_preprocess.py: Methods of data processing and heterogeneous graph construction
- metric.py: Evaluation metrics calculation (AUC, AUPR, F1-score, etc.)
- model.py: GHTMDA model architecture with hierarchical attention mechanism
- train.py: Training script for metabolite-disease association prediction

<img width="4383" height="2475" alt="overall architecture" src="https://github.com/user-attachments/assets/98297c55-c721-4322-893b-3bea56dff943" />
