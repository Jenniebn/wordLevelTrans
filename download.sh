#!/bin/bash

# Create data directory and download necessary data
(
mkdir data
cd data

# Download individual data files from Google Drive
gdown --folder "https://drive.google.com/drive/folders/1PRyJirZIPe4Rd5VT5R1gggG3vmU3rnz0?usp=sharing" -O .

cd ..
)
(
mkdir ckpt
cd ckpt

gdown https://drive.google.com/uc?id=1-671w3Re144-6UxCAQRg3nKB9_UmXiOV
gdown https://drive.google.com/uc?id=1-2gLisBygm5c9CukQcL8YcYwgx1limXC

cd ..
)
