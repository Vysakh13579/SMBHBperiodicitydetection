# SMBHB Periodicity Detection

This repository contains tools and scripts for Supermassive Black Hole Binary (SMBHB) periodicity detection.

## Prerequisites
Since we are using pioran(old version), This project require packages that a few versions behind. please use the requirements file for reproducibility.

* **Python 3.10**: This project requires Python 3.10. Please ensure you have it installed before proceeding.

> **Note:** It is strongly recommended to run this project inside a virtual environment (e.g., `venv` or `conda`), as some of the required packages are older and might conflict with your system-wide Python installation.

## Installation


```bash
pip install -r requirements.txt
pip install .

```
root/
├── AGNobsdata/
├── helios_files/
├── json_files/
├── my_utils.egg-info/
├── old_files/
├── pioran/
├── real_data_tests/
├── simDATAcsvs/
├── utils/
│   ├── NSmodels_FERMI.py
│   ├── NSmodels_graham.py
│   ├── NSmodels.py
│   ├── NSmodels2.py
│   ├── plotter.py
│   └── THESIS.py
├── alogrithm_batch_run.py
├── batch_run.py
├── example.ipynb
├── real_data_test_analysis.ipynb
├── requirements.txt
├── sim_data_maker.ipynb
├── thesis_plot.ipynb
└── tk_lc_simulator.ipynb