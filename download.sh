#!/bin/bash

# Create data directory and download necessary data
(
mkdir -p data
cd data

# Download individual data files from Google Drive
gdown --folder "https://drive.google.com/drive/folders/1PRyJirZIPe4Rd5VT5R1gggG3vmU3rnz0?usp=sharing" -O .

cd ..
)
(
mkdir -p ckpt
cd ckpt

gdown https://drive.google.com/file/d/1-2gLisBygm5c9CukQcL8YcYwgx1limXC/view?usp=drive_link
gdown https://drive.google.com/file/d/1-671w3Re144-6UxCAQRg3nKB9_UmXiOV/view?usp=drive_link

cd ..
)
