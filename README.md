# Aligning Embedding Spaces Across Languages to Identify Word Level Equivalents in the Context of Concreteness and Emotion
Josephine Kaminaga, Jennie Wu, Daniel Yeung

This research developed a neural network model using relative word embeddings to investigate the impacts of emotionality and abstract on a bilingual semantic space mapping. Our model achieved a maximum of 2.38% accuracy, with emotion-laden and concrete words having the greatest accuracy. An in-depth error analysis revealed that although the model didn't learn a word-to-word mapping, it generally achieved a mapping of sub-regions onto each other, with a handful of errors being due to a lack of data and cultural differences impacting word representations. The model's performance agrees with the trend in literature of emotional and concrete words providing a processing advantage, and furthermore suggests that this processing advantage is cross-lingual.

## Folder Documentation

The Cambridge dictionary JSON and scraper folders, as well as the EN-ZH, Facebook AI, MDBG, and Yabla folders, contain the Python scripts used to collect the English-Mandarin translation data from various online dictionaries. They also contain the individual and combined JSON files with the processed data. The golden set folder contains the Python code used to build the final translation dictionary we used for model training, and the translation dictionary itself. Please do not run the scripts in these folders as they each request around 100k webpages from various online dictionaries and take around 8 hours to fully execute.
