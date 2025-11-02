# MedNLP - Medical Condition and Drug Recommendation System

A machine learning project that analyzes drug reviews to predict medical conditions and recommend appropriate medications based on patient reviews and symptoms.

## Overview

This project uses Natural Language Processing (NLP) and machine learning techniques to:
- Classify medical conditions based on drug reviews
- Recommend appropriate medications for specific conditions
- Analyze patient feedback and drug effectiveness

## Dataset

The project uses drug review data from Drugs.com containing:
- **drugsComTrain.tsv**: Training dataset with drug reviews, conditions, ratings, and metadata
- **drugsComTest.tsv**: Test dataset for model evaluation
- **conditions.tsv**: Medical condition descriptions and mappings

### Key Features:
- Drug names and medical conditions
- Patient reviews and ratings (1-10 scale)
- Review dates and usefulness counts
- 161,297 total reviews across multiple conditions

## Machine Learning Models

The project implements and compares several classification algorithms:

1. **Multinomial Naive Bayes (MultinomialNB)**
   - With Count Vectorization
   - With TF-IDF Vectorization

2. **Passive Aggressive Classifier**
   - With Count Vectorization
   - With TF-IDF Vectorization
   - With N-gram features (1,2)

## Text Processing

- **Feature Extraction**: Count Vectorization and TF-IDF
- **Preprocessing**: Stop words removal, text normalization
- **N-gram Analysis**: Unigrams and bigrams for better context understanding

## Key Medical Conditions Analyzed

The project focuses on four major conditions:
- **Birth Control** (28,788 reviews)
- **Depression** (9,069 reviews)
- **High Blood Pressure** (2,321 reviews)
- **Diabetes, Type 2** (2,554 reviews)

## Exploratory Data Analysis

- Word cloud visualizations for different medical conditions
- Distribution analysis of conditions and ratings
- Review text analysis and pattern identification

## Requirements

```python
pandas
numpy
scikit-learn
matplotlib
seaborn
wordcloud
```

## Usage

1. Load and preprocess the drug review data
2. Extract features using TF-IDF or Count Vectorization
3. Train classification models on the processed data
4. Evaluate model performance using accuracy, precision, and recall
5. Generate predictions for new drug reviews

## Model Performance

The project evaluates models using:
- Accuracy scores
- Confusion matrices
- Classification reports
- Cross-validation results

## Files Structure

```
├── Condition and drug reco.ipynb    # Main analysis notebook
├── drugsComTrain.tsv               # Training dataset
├── drugsComTest.tsv                # Test dataset
├── conditions.tsv                  # Condition descriptions
└── README.md                       # Project documentation
```

## Applications

This system can be used for:
- **Healthcare Analytics**: Understanding drug effectiveness patterns
- **Medical Research**: Analyzing patient feedback trends
- **Drug Recommendation**: Suggesting medications based on conditions
- **Sentiment Analysis**: Evaluating patient satisfaction with treatments

## Future Enhancements

- Deep learning models (LSTM, BERT) for better text understanding
- Real-time drug recommendation API
- Integration with medical databases
- Multi-language support for global healthcare applications

## Contributing

Feel free to contribute to this project by:
- Improving model accuracy
- Adding new features
- Enhancing data preprocessing
- Creating better visualizations

## License

This project is open source and available under the MIT License.