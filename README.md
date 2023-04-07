# Deep Learning for Encrypted Network Traffic Classification: Exploring New Techniques for Efficient Traffic Management - A Comparative Analysis

This repository contains the code for the a research project investigating the performance of various machine learning and deep learning models for encrypted traffic classification based on TLS handshake exchange. The primary goal is to determine the effectiveness of these models in classifying application traffic operating over encrypted channels.

## Models

The following models are included in this repository:

1. **1D-CNN**: One-dimensional Convolutional Neural Network
2. **ADT**: AdaBoost Decision Trees
3. **BD-GRU**: Bidirectional Gated Recurrent Unit
4. **BD-LSTM**: Bidirectional Long Short-Term Memory
5. **RB-RF**: Recomposed Bytes with Random Forest (control model from the original work of D. Shamsimukhametov et al., 2022)
6. **Ensemble Models**: Combination of selected models for improved performance

Each model has its own Python file, which includes the implementation, training, and evaluation of the respective model.

## Repository Structure

.<br>
├── 1D-CNN.py<br>
├── ADT.py<br>
├── BD-GRU.py<br>
├── BD-LSTM.py<br>
├── RB-RF.py<br>
├── ensemble.pyMbr>
├── Kieran Brooks CS435 Final Report.pdf<br>
└── README.md<br>

## Usage

To run the code for each model, use the following commands:

python 1D-CNN.py
python ADT.py
python BD-GRU.py
python BD-LSTM.py
python RB-RF.py
python ensemble.py

## Dependencies

Please ensure you have the following Python packages installed:

- numpy
- pandas
- scikit-learn
- tensorflow
- keras

You can install these packages using `pip`: 

pip install numpy pandas scikit-learn tensorflow keras

## Citation

If you find this repository useful, please cite the following paper:

- (Brooks, K. (2023). Deep Learning for Encrypted Network Traffic Classification: Exploring New Techniques for Efficient Traffic Management - A Comparative Analysis. CS 435 - Dr. Qian Yu. [GitHub Repository]. Retrieved from https://github.com/kierankyllo/tls_traffic_analysis)

## License

This project is licensed under the MIT License.
