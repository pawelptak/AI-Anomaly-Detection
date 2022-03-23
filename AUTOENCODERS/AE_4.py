from Classification.AutoencoderResultsClassificator import AutoencoderResultsClassificator
from DataPreparing.KDDLoader import KDDLoader
from DataPreparing.KDDPreprocessor import KDDPreprocessor
from Execution.AutoencoderModelExecutor import AutoencoderModelExecutor
from Model.AutoEncoderModelDense import AutoEncoderModelDense_1, AutoEncoderModelDense_2
import numpy as np
import matplotlib.pyplot as plt

neptune = np.fromfile("Results/AE_2_results/neptune_scores.npy")
back = np.fromfile("Results/AE_2_results/back_scores.npy")

# print(neptune)
print(back)
