#!/bin/bash

# Create a directory and download pretrained checkpoints
(
cd enzh_encoder_decoder
mkdir -p data
cd data

cd enzh_encoder_decoder
mkdir -p data
cd data 
cd ..
mkdir -p ckpt
cd ckpt
cd ..

# Example individual files from Google Drive
gdown https://drive.google.com/uc?id=1norNWWGYP3EZ_o05DmoW1ryKuKMmhlCX
gdown https://drive.google.com/uc?id=1QEl-btGbzQz6IwkXiFGd49uQNTUtTHsk

# Your shared folder (model checkpoints)
gdown --folder "https://drive.google.com/drive/folders/1D4uUwDci1jXn8SWadlKABxRGgJuKo1np" -O .
)

# Download data and extract it
(
gdown https://drive.google.com/uc?id=1Q_dxuyI41AAmSv9ti3780BwaJQqwvwMv
unzip data.zip
rm data.zip
)
