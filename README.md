# Personality‑Integrated Large Language Models for Discriminating Suicidal Ideation from Supportive Contents on Social Media

# Project Overview

Recent studies have applied transformer-based models for suicide ideation detection on social media. However, many existing datasets label all suicide-related content as suicidal ideation without distinguishing user intent.

As a result, supportive, preventive, and awareness-oriented posts are frequently misclassified as suicidal, leading to high false positive rates and reduced system reliability.

This project explores three main approaches to improve intent-sensitive suicide ideation detection:

- Dataset relabeling to reduce label noise
- Personality trait integration using Big Five (OCEAN) features
- Natural Language Inference (NLI) filtering to capture semantic intent

---

# Related Resources
The final trained models, datasets, and interactive demo are hosted separately on Hugging Face.

## Hugging Face Resources

- Dataset: https://huggingface.co/datasets/lensy111/relabelled-suicide-ideation
- Demo: https://huggingface.co/spaces/lensy111/roberta_suicide_ideation_personality_nli

---

# Datasets

## Suicide and Depression Detection Dataset

Source: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch

- 232,074 Reddit posts collected from 2008–2021
- `r/SuicideWatch` posts labeled as suicidal
- `r/teenagers` posts labeled as non-suicidal

### Identified Limitation

The original dataset assigns labels based solely on subreddit origin, causing supportive, preventive, and awareness-related posts to be incorrectly labeled as suicidal ideation.

This introduces label noise and contributes to high false positive rates.

### Dataset Subsets

The following subsets were used during experimentation:

- 2,000 posts for relabeling and training experiments
- 1,000 posts reserved for evaluation
- Full dataset version excluding the evaluation split

---

## False Positive Evaluation Dataset

Source: https://zenodo.org/records/2667859

- 500 anonymized Reddit posts annotated using the Columbia Suicide Severity Rating Scale (C-SSRS)
- 89 supportive samples selected for cross-dataset false positive evaluation
- Used exclusively for evaluating supportive-content misclassification

---

# Methodology

## 1. Dataset Relabeling

A semi-automated relabeling pipeline was developed to identify potentially mislabeled supportive posts.

### Relabeling Workflow

1. Manual curation of a high-quality subset
2. Sentiment threshold analysis using `cardiffnlp/twitter-roberta-base-sentiment`
3. Soft-label scoring on the full dataset
4. Human-in-the-loop verification of flagged posts

### Relabeling Results

- Potential false positives flagged: 4,039 posts
- Confirmed supportive false positives: 1,739 posts

The relabeling process substantially reduced label noise in supportive and awareness-oriented content.

---

## 2. Text Preprocessing

Preprocessing steps included:

- URL and irrelevant symbol removal
- Quote and apostrophe normalization
- Emoji-to-text conversion
- Punctuation and whitespace normalization
- Sentence boundary correction

---

## 3. Personality Feature Integration

Big Five personality traits were extracted using:

`ppp57420/ocean-personality-distilbert`

Selected traits:

- Neuroticism
- Extraversion
- Agreeableness

Personality features were integrated with transformer embeddings to improve contextual understanding of user intent.

---

## 4. Transformer Models

The following transformer architectures were evaluated:

- BERT
- RoBERTa
- Mental-RoBERTa

Configurations tested:

- Original vs relabeled datasets
- With and without personality features
- With and without NLI filtering

### Personality Feature Integration

CLS embeddings were concatenated with personality trait vectors and passed through a hidden layer:

- Hidden size: 128
- Activation: ReLU
- Dropout: 0.3

---

## 5. Natural Language Inference (NLI) Filtering

NLI filtering was applied as a post-processing step to reduce false positives.

### NLI Models Used

- `tasksource/deberta-small-long-nli`
- `tasksource/deberta-base-long-nli`

### Approach

Predicted suicidal posts were paired with supportive or awareness-oriented hypotheses such as:

> “The author discusses suicide in general awareness or prevention, not personal suicidal thoughts.”
> “The post shares support, resources, or hotlines, not the author's own suicidal intent.”
> “The author reflects to inspire or thank others, not describing current suicidal thoughts.”

Entailment probabilities were computed using pretrained NLI models.

Posts exceeding the selected threshold were relabeled as non-suicidal.

### Threshold Optimization

Thresholds were selected to balance:

- F1 performance
- False Positive Rate (FPR)

---

# Evaluation Metrics

Primary metrics:

- Accuracy
- Precision
- Recall
- Macro F1-score

Additional focus:

- False Positive Rate (FPR)

FPR was evaluated using supportive samples from the Reddit C-SSRS dataset to measure supportive-content misclassification.

---

# Results

## Impact of Dataset Relabeling

| Model | Dataset | F1 | Accuracy | FPR |
|---|---|---|---|---|
| BERT | Original | 0.90 | 90% | 83.15% |
| BERT | Relabeled | 0.92 | 92% | 35.96% |
| RoBERTa | Original | 0.91 | 91% | 88.76% |
| RoBERTa | Relabeled | 0.92 | 92% | 53.93% |
| Mental-RoBERTa | Original | 0.89 | 89% | 93.26% |
| Mental-RoBERTa | Relabeled | 0.94 | 94% | 28.09% |

Dataset relabeling significantly improved both classification performance and false positive reduction.

---

## Impact of Personality Features and NLI

Best-performing configuration:

| Model | Features | F1 | Accuracy | FPR |
|---|---|---|---|---|
| RoBERTa | Personality + NLI | 0.94 | 94% | 14.61% |

Key observations:

- Personality features improved RoBERTa stability and balance
- NLI filtering reduced supportive-content false positives
- Combined personality + NLI integration achieved the best overall balance

---

# Key Findings

- Dataset quality significantly impacts suicide ideation detection performance
- Supportive suicide-related posts are a major source of false positives
- Personality traits can improve contextual modeling
- NLI filtering helps distinguish supportive vs self-directed suicidal intent
- RoBERTa with personality + NLI achieved the best overall performance

---

# Limitations

- Dataset sourced from only two Reddit subreddits
- Personality traits inferred from individual posts may be noisy
- Only one NLI hypothesis set was evaluated
- Non-suicidal posts were not systematically relabeled

---

# Future Work

Potential future improvements include:

- Expanding to additional social media platforms
- User-level behavioral modeling
- Alternative NLI hypothesis generation strategies
- Evaluation on larger and more diverse datasets
- Comparison with additional transformer architectures

---

# Research Context

This project was conducted as part of a Master of Artificial Intelligence research study at Universiti Malaya focused on improving reliability and intent-awareness in suicide ideation detection systems.
