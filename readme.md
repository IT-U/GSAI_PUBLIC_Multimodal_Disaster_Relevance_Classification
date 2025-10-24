# Replication code for: A Multimodal GeoAI Approach to Combining Text with Spatiotemporal Features for Enhanced Relevance Classification of Social Media Posts in Disaster Response

This repository contains the replication code and materials for the study:

- Hanny et al. (2025): [A multimodal GeoAI approach to combining text with spatiotemporal features for enhanced relevance classification of social media posts in disaster response](https://www.tandfonline.com/doi/full/10.1080/20964471.2025.2572140#abstract)

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

Due to Twitter’s (now X) [API terms](https://developer.x.com/en/developer-terms/agreement-and-policy), we are unable to share full tweet content. However, we provide tweet IDs, ground truth relevance labels and our engineered spatiotemporal features. These can be accessed and rehydrated using the X v2 API. This dataset and accompanying metadata are available on [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0DBK04).

## 📖 Citation

If you use this code or dataset in your research, please cite our work accordingly.
```bibtex
@article{Hanny.2025c,
  title = {A Multimodal {{GeoAI}} Approach to Combining Text with Spatiotemporal Features for Enhanced Relevance Classification of Social Media Posts in Disaster Response},
  author = {Hanny, David and Schmidt, Sebastian and Gandhi, Shaily and Granitzer, Michael and Resch, Bernd},
  year = {2025},
  journal = {Big Earth Data},
  volume = {0},
  number = {0},
  pages = {1--45},
  publisher = {Taylor \& Francis},
  issn = {2096-4471},
  doi = {10.1080/20964471.2025.2572140},
  urldate = {2025-10-24},
  abstract = {Geo-referenced social media data supports disaster management by offering real-time insights through user-generated content. To identify critical information amid high volumes of noise, classifying the relevance of posts is essential. Most existing methods primarily use textual features, neglecting spatial and temporal context despite its importance in determining relevance. This study proposes a multimodal approach that integrates text with spatiotemporal features for relevance classification of geo-referenced social media posts. We evaluate our method on 4,574 manually labelled posts from five disasters: the 2020 California wildfires, 2021 Ahr Valley floods, 2023 Chile wildfires, 2023 Turkey earthquake and 2023 Emilia-Romagna floods. Labels were assigned based on text, geographic location and time. Our spatiotemporal features include proximity to disaster impact sites, local co-occurrences with disaster-related posts, event type and geographic context. When utilised on their own, they achieved a macro F1 score of 0.713 with a random forest classifier. A fine-tuned TwHIN-BERT-base model using only text scored 0.779. For multimodal classification, we tested feature concatenation, in-context learning, stacking and partial stacking. Partial stacking produced the highest macro F1 score (0.814). Our multilingual, context-aware classification approach lays the groundwork for more integrated GeoAI applications in disaster management, the social sciences and beyond.},
  keywords = {disaster management,GeoAI,Machine learning,multimodal learning,Published,relevance classification,social media}
}
```

## 🛠 Contact
In case of questions, please contact: 
David Hanny (david.hanny@it-u.at), 
IT:U Interdisciplinary Transformation University Austria


