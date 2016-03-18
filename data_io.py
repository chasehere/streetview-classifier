import csv, json, os, pickle
import numpy as np
import pandas as pd

def get_paths():
  paths = json.loads(open('settings.json').read())
  for key in paths:
    paths[key] = os.path.expandvars(paths[key])
  return paths

def get_2016_tax_sale_list():
  paths = get_paths()
  return pd.read_csv(paths['2016_tax_sale_list'])
