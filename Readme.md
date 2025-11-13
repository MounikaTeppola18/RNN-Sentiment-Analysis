
# RNN Sentiment Classification Project

## Overview
This project implements and evaluates multiple Recurrent Neural Network (RNN) architectures for **sentiment classification** on the IMDb Movie Review dataset.  
You can experiment with **RNN, LSTM and Bidirectional LSTM** architectures and different hyperparameter combinations.

---

## Folder Structure
```

RNN-Sentiment-Classification/
├── data/                 
│   ├──imdb_seq25.npz
│   ├──imdb_seq50.npz
│   ├──imdb_seq100.npz              
├── src/
│   ├── preprocess.py      # Preprocessing script
│   ├── env_info.py        # System info 
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   ├──src/
|   │   └── results_plot.ipynb # Visualization notebook
├── results/
│   ├── metrics.csv        # Results for all model combinations
│   ├── summary_table.csv 
│   └── plots/             # Generated graphs
├── report.pdf             # Project report
├── requirements.txt       # Dependencies
├── README.md
└── .gitignore

````

---

## Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/<your-username>/RNN-Sentiment-Classification.git
cd RNN-Sentiment-Classification
````

2. **Install required Python libraries**:

```bash
pip install -r requirements.txt
```

3. **Prepare the dataset**:

* The dataset link is given in the References section.

* Download it and place it in the data folder before running any scripts.

* Make sure the dataset remains inside the `data/` folder.

---

## Preprocessing

* Open `src/preprocess.py` and **replace `data_path` in the main function** with the path to your unzipped dataset in the `data/` folder.
* To run preprocessing:

```bash
cd src
python preprocess.py
```

---

## Environment Information

* To check your system info (CPU/GPU, RAM), run:

```bash
python src/env_info.py
```

---

## Training the Models

* To train models, go to `train.py` and use the `--architecture` argument:

```bash
python src/train.py --architecture rnn
```

* Supported architectures: `rnn`, `lstm`, `bilstm`.
* Running `rnn` will execute **all 18 RNN combinations**. Similarly for `lstm` and `bilstm`.

---

## Evaluation & Visualization

1. **Visualization Notebook**:

   * Open `src/results_plot.ipynb`
   * Run all cells to generate graphs.

2. **Results**:

   * Metric summaries are in `results/metrics.csv`.
   * Generated plots are in `results/plots/`.
   * General summary tables are in the `results/` folder.

---

## Expected Runtime and Output Files

- **Expected Runtime**: Running all model combinations on CPU may take **approximately 2 hours** for 54 combinations.  

- **Output Files**:

| File/Folder | Description |
|-------------|-------------|
| `results/metrics.csv` | Contains accuracy, F1-score, and other metrics for all model combinations. |
| `results/plots/` | Contains generated graphs for accuracy/F1 vs. sequence length and training loss over epochs. |
| `src/results_plot.ipynb` | Notebook for visualizing model performance and generating plots. |


## Notes

* Make sure the dataset is in the `data/` folder before running any scripts.
* Always run scripts from the `src/` directory for correct relative paths.

---

## References

* https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
* Python libraries used are listed in `requirements.txt`.


