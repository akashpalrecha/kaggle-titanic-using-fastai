#%%
import numpy as np
import pandas as pd
from pathlib import Path

#%%
PATH = Path('titanic/test.csv')
df = pd.read_csv(PATH)

#%%
replace_nan = lambda x: df.Fare.median() if pd.isnull(x) else x
df.Fare = df.Fare.apply(replace_nan)

#%%
df.to_csv('titanic/processed_test.csv', index=False)

#%%
