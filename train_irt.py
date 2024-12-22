# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
import irt
import utils

random_state = 42

# %%

data = pd.read_csv("data/merged_df.csv")
print(data.head())

data_scores = data.drop(columns=["id"]).fillna(0).values
print(data_scores.shape)
print(data_scores)

# %%

# D = 10 # Dimensions to try
# device = 'cpu' # Either 'cuda' or 'cpu' 
# epochs = 100  # Number of epochs for IRT model training (py-irt default is 2000)
# lr = .1  # Learning rate for IRT model training (py-irt default is .1)

# # Saving the training dataset in the needed format
# irt.create_irt_dataset(data_scores, 'data/irt_val_dataset.jsonlines')

# # Trying different Ds

# dataset_name = 'data/irt_val_dataset.jsonlines'
# model_name = 'data/'

# # Load trained IRT model parameters
# irt.train_irt_model(dataset_name, model_name, D, lr, epochs, device)
#A, B, Theta = irt.load_irt_parameters(model_name)
# %%

from py_irt.dataset import Dataset
from py_irt.models import OneParamLog
import pandas as pd
# %%

import py_irt
# %%
