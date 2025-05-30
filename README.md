# Aligning Embedding Spaces Across Languages to Identify Word Level Equivalents in the Context of Concreteness and Emotion
Josephine Kaminaga, Jennie Wu, Daniel Yeung

<p align="center">
    <img alt="architecture" src="./static/architecture.jpg">
</p>

# Setup
1. Code
```
git clone https://github.com/Jenniebn/wordLevelTrans.git
```
2. Environment
```
pip install numpy torch tqdm 
```
3. Download models and data
```
pip install --upgrade gdown && bash ./download.sh
```
This downloads all the preprocessed data and model checkpoint.

4. Run notebooks

# GitHub Contents

- `dictionary/`: A directory containing all the scripts needed to scrape and process the 4 online dictionaries (Cambridge, MDBG, Yabla, and Facebook AI) we used.

    - `dictionary/golden_set`: A directory containing the Python code used to combine the four individual translation dictionaries into the full, final dictionary we used for model training, and the translation dictionary itself.
    - `dictionary/*/* Scrapers`: Directories containing the Python code used to scrape online dictionaries.
    - `dictionary/*/JSON Data`: A directory containing the unique English-Mandarin translations scraped from that dictionary. 

- `code/`: A directory containing the Python code to train the Chinese to Chinese Autoencoder as well as the English to Chinese Encoder Decoder model.

# Note
To run the scripts in `dictionary/`, please expect taking around 8 hours to fully execute as they each request around 100k webpages from various online dictionaries.

# BibTeX
```bibtex
@inproceedings{kaminaga-wu-yeung-2025-embed,
    title={Aligning Embedding Spaces Across Languages to Identify Word Level Equivalents in the Context of Concreteness and Emotion},
    author={Kaminaga, Josephine and Wu, Jingyi and Yeung, Daniel{\`a}},
    booktitle = "Proceedings of the Society for Computation in Linguistics 2025",
    month = jun,
    year = "2025",
    publisher = "Association for Computational Linguistics",
    pages = "1--9"
}
```

# Acknowledgement
This repository is based on [relreps](https://github.com/lucmos/relreps?tab=readme-ov-file) from Moschella et al.