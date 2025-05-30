# Aligning Embedding Spaces Across Languages to Identify Word Level Equivalents in the Context of Concreteness and Emotion
Josephine Kaminaga, Jennie Wu, Daniel Yeung

This research developed a neural network model using relative word embeddings to investigate the impacts of emotionality and abstractness on a bilingual semantic space mapping. Our model achieved a maximum of 14.36% accuracy, with emotional and concrete words having the greatest accuracy. An in-depth error analysis revealed that although the model didn't learn a word-to-word mapping, it generally achieved a mapping of sub-regions onto each other, with a handful of errors being due to a lack of data and cultural differences impacting word representations. The model's performance agrees with the trend in literature of emotional and concrete words providing a processing advantage, and furthermore suggests that this processing advantage is cross-lingual.

# GitHub Contents

- `dictionary/`: A directory containing all the scripts needed to scrape and process the 4 online dictionaries (Cambridge, MDBG, Yabla, and Facebook AI) we used in this project.

    - `dictionary/golden_set`: A directory containing the Python code used to combine the four individual translation dictionaries into the full, final dictionary we used for model training, and the translation dictionary itself.
    - `dictionary/*/* Scrapers`: Directories containing the Python code used to scrape online dictionaries.
    - `dictionary/*/JSON Data`: A directory containing the unique English-Mandarin translations scraped from that dictionary. 

- `code/`: A directory containing the Python code to train the Chinese to Chinese Autoencoder as well as the English to Chinese Encoder Decoder model.
   

# Note
To run the scripts in `dictionary/`, please expect taking around 8 hours to fully execute as they each request around 100k webpages from various online dictionaries.
