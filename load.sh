#!bin/bash

echo "Downloading images..."
wget https://github.com/Sosnovik/u-rocket/releases/download/v0.0/photo.zip
unzip photo.zip 
rm photo.zip

echo "Downloading models..."
wget https://github.com/Sosnovik/u-rocket/releases/download/v0.0/zf_unet_224.h5

if [ ! -d "$saved_models" ]; then
  mkdir saved_models
fi

mv zf_unet_224.h5 saved_models

