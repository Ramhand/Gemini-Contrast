# Gemini-Contrast

Gemini-Contrast is a novel deep learning architecture designed for efficient
image classification.

# Features

- A combination of Supervised and Unsupervised Contrastive Learning methods
  to pack as much information into the feature space as possible.

- Scalable to large datasets with efficient memory usage

# Installation
```bash
git clone https://github.com/Ramhand/Gemini-Contrast
cd Gemini-Contrast
pip install -e .
```
To train your model, run:
```bash
python train.py --data_root path/to/your/data --epochs 50 --batch_size 64 --learning_rate 1e-4 --use_gpu
```
The first part of training is getting Castor and Pollux up and running through the prework method, but after that, they will both train with the Classification Head to
ensure highest accuracy.
