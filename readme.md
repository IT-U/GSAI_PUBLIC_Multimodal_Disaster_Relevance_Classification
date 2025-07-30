# Replication code for: A Multimodal GeoAI Approach to Combining Text with Spatiotemporal Features for Enhanced Relevance Classification of Social Media Posts in Disaster Response

This repository contains the replication code and materials for the study:

**Hanny, D., Schmidt, S., Gandhi, S., Granitzer, M., & Resch, B. (2025).**  
*A Multimodal GeoAI Approach to Combining Text with Spatiotemporal Features for Enhanced Relevance Classification of Social Media Posts in Disaster Response.*  
[Submitted to *Big Earth Data*]


## 📄 Overview

This study presents a multimodal relevance classification approach that integrates textual, spatial, and temporal features to improve relevance classification of social media posts in natural disaster scenarios. The pipeline includes:

- Pre-processing of geo-referenced tweets
- Feature engineering and evaluation (spatial, temporal, co-occurrence)
- Non-text classifier training evaluation
- Text classifier (TwHIN-BERT) training and evaluation
- Multimodal fusion with feature concatenation, partial stacking and in-context learning
- Model comparison, visualisation and explanation with [SHAP](https://shap.readthedocs.io/en/latest/)

## 📁 Repository Structure
```
├── environment.yml # Conda environment file
├── notebooks/      # Numbered notebooks for each processing stage
│ ├── 01_ahr_valley.ipynb
│ ├── 02_label_preparation.ipynb
│ ├── ...
│ ├── 07_meta_learning.ipynb
│ └── 08_visualisation.ipynb
├── src/ # Helper modules and reusable functions
│ ├── model_training/
│ │ ├── bert.py                 # standard BERT fine-tuning
│ │ ├── classification_head.py  # non-text classification and meta-learning
│ │ ├── extended_bert.py        # BERT extended with spatiotemporal features
│ ├── utils.py      # helper functions
│ └── ...
└── readme.md # Project documentation
```

Each notebook corresponds to a specific step in the pipeline, from data loading and preprocessing to model training, inference, and evaluation. Reusable functions are available in the `src/` directory.

## ⚙️ Getting Started

### 1. Create environment

To replicate the experiments, create the conda environment:

```bash
conda env create -f environment.yml
conda activate relevance-classification
```

### 2. Run notebooks

Execute the numbered notebooks (`notebooks/01_...` to `09_...`) in order. Each notebook is self-contained and documented.

## 📊 Data Availability

Due to Twitter’s (now X) [API terms](https://developer.x.com/en/developer-terms/agreement-and-policy), we are unable to share full tweet content. However, we provide tweet IDs, ground truth relevance labels and our engineered spatiotemporal features. These can be accessed and rehydrated using the X v2 API. This dataset and accompanying metadata are available on [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0DBK04)

## 📖 Citation

If you use this code or dataset in your research, please cite our work accordingly.
```bibtex
@article{hanny2025multimodalRelevance,
  title     = {A Multimodal GeoAI Approach to Combining Text with Spatiotemporal Features for Enhanced Relevance Classification of Social Media Posts in Disaster Response},
  author    = {Hanny, David and Schmidt, Sebastian and Gandhi, Shaily and Granitzer, Michael and Resch, Bernd},
  journal   = {TBD},
  year      = {2025}
}
```

## 🛠 Contact
In case of questions, please contact: 
David Hanny (david.hanny@it-u.at), 
IT:U Interdisciplinary Transformation University Austria