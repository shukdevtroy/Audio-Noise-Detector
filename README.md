# Audio-Noise-Detector

---

## UrbanSound8K Audio Classification

This repository contains code for preprocessing audio data from the UrbanSound8K dataset, training a Convolutional Neural Network (CNN) for audio classification, and making predictions on new audio files. The repository includes three main scripts:

1. `preprocess2.py`: Preprocesses audio files and extracts features.
2. `train.py`: Trains a CNN model on the extracted features and evaluates its performance.
3. `test2.py`: Loads a trained model and makes predictions on new audio files.

## Prerequisites

Before running the scripts, make sure you have the following packages installed:

- `pandas`
- `numpy`
- `scipy`
- `soundfile`
- `python_speech_features`
- `keras`
- `tqdm`

You can install these dependencies using pip:

```bash
pip install pandas numpy scipy soundfile python_speech_features keras tqdm
```

## Data Preparation

1. **Dataset**: You need the UrbanSound8K dataset. Place the dataset in the following directory:
   ```
   F:/Rashmama office/noise github/UrbanSound8K/UrbanSound8K/audio/fold
   ```
   Make sure the dataset is in the correct format with audio files organized into folds.

2. **Metadata File**: Ensure that the metadata CSV file `UrbanSound8K.csv` is located at:
   ```
   F:/Rashmama office/Denoiser-master/Denoiser-master/compressed_dataset/UrbanSound8K.csv
   ```

## Scripts

### 1. `preprocess2.py`

This script preprocesses audio files from the UrbanSound8K dataset to extract features and save them in CSV format.

**Usage**:
```bash
python preprocess2.py
```

**Description**:
- Reads metadata from `UrbanSound8K.csv`.
- Extracts MFCC (Mel-frequency cepstral coefficients) and filter bank features from each audio file.
- Saves the processed features and labels to CSV files (`train_data.csv`, `test_data.csv`, `train_labels.csv`, and `test_labels.csv`).

### 2. `train.py`

This script trains a CNN model on the extracted features and evaluates its performance.

**Usage**:
```bash
python train.py
```

**Description**:
- Loads the training and testing data from CSV files generated by `preprocess2.py`.
- Converts labels to one-hot encoding.
- Reshapes data for CNN input.
- Defines and compiles a CNN model.
- Trains the model and evaluates its performance.
- Saves the trained model to `model.h5`.

### 3. `test2.py`

This script loads a trained model and makes predictions on new audio files.

**Usage**:
```bash
python test2.py
```

**Description**:
- Loads a pre-trained model from `model.h5`.
- Prompts the user for the path to an audio file.
- Extracts features from the provided audio file.
- Makes predictions using the loaded model.
- Prints the top 3 predicted classes.

## File Structure

```
/UrbanSound8K
  /audio
    /fold1
    /fold2
    ...
  UrbanSound8K.csv
/preprocess2.py
/train.py
/test2.py
```

## Class Labels

The following class labels are used for classification:

- 0: Windy
- 1: Horn
- 2: Children-noise
- 3: Dog Bark
- 4: Drilling
- 5: Engine Idling
- 6: Gun Shot
- 7: Jackhammer
- 8: Siren
- 9: Street music

## Notes

- Ensure all paths in the scripts are correctly set according to your directory structure.
- The model architecture and hyperparameters in `train.py` can be adjusted based on your needs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please open an issue on this GitHub repository or contact [Shukdev Datta] at [shukdevdatta@gmail.com] for UrbanSound8k dataset.

---

Feel free to adjust any details or add additional information specific to your needs.
