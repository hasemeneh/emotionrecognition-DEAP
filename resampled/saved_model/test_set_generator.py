import pandas as pd
import numpy as np
csvArousal = pd.read_csv('intersectedXarousal_set.csv')
csvValence = pd.read_csv('intersectedXvalence_set.csv')
key = csvValence.columns.values[0]
intersected_set = csvArousal.merge(csvValence,left_on=key,right_on=key,how='inner')
intersected_set.to_csv('intersected_set.csv')