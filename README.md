# Bioacoustic Classifier – MSc Dissertation Project

This repository accompanies my MSc dissertation, which investigates how **Active Learning (AL)** and **Pretraining with Convolutional Autoencoders (CAEs)** can reduce the manual annotation burden in **Passive Acoustic Monitoring (PAM)**.  
It contains the full experimental pipeline, from raw recordings through spectrogram generation, model training, and evaluation.

---

## Project Overview
Passive Acoustic Monitoring provides vast amounts of biodiversity data but is limited by the effort required for manual annotation.  
This project explores two strategies to reduce annotation effort:
- **Active Learning (AL):** selecting the most informative samples for annotation.  
- **Pretraining (CAE):** learning feature representations from unlabelled audio.  

Models were trained on **log-mel spectrograms** of bird vocalisations using CNNs, and evaluated with **F1, Recall, Precision, and Hamming Loss**.

---

## Repository Structure
- **data/**  
  - `raw/` – raw audio recordings  
  - `processed/` – spectrograms and processed chunks  
  - `annotations/` – label files  

- **scripts/**  
  - `generate_spectrograms.py` – convert audio to spectrograms  
  - `preprocess.py` – preprocessing pipeline  
  - `active_learning_experiment.py` – runs AL vs random sampling  
  - `CAE Implementation.ipynb` – autoencoder pretraining experiments  

- **src/** – core source code (modules/utilities)  

- **utils.py** – helper functions  
- **requirements.txt** – Python dependencies  
- **setup.py** – install script  
- **LICENSE**  
- **README.md**  

---

## Installation
Set up a Python environment (Python 3.10 recommended) and install dependencies:

```bash
pip install -r requirements.txt
