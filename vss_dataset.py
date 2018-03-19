import pandas as pd
import glob
import os
import torch
from torch.utils.data import Dataset, Dataloader
import numpy as np

class vss_dataset(Dataset):
	def __init__(self, csv_path):
	"""
		Args
			csv_file(string): Path to the csv file
	"""
		filenames = glob.glob(os.path.join(csv_path, "*.csv"))
		df = (pd.read_csv(f, skiprows = 2) for f in filenames)
		self.data = pd.concat(df)

		self.rb1 = self.data[['R1X','R1Y','R1T']]
		self.rb2 = self.data[['R2X','R2Y','R2T']]
		self.rb3 = self.data[['R3X','R3Y','R3T']]

		self.adv1 = self.data[['A1X','A1Y','A1T']]
		self.adv2 = self.data[['A2X','A2Y','A2T']]
		self.adv3 = self.data[['A3X','A3Y','A3T']]

		self.ball = self.data[['BX','BY']]

		self.time = self.data['TIMESSTAMP_FRAME']
		self.df_index = self.data.index.values

	def __len__(self):
		return len(self.df_index)

	def __getitem__(self, idx):
		return (self.data.iloc[idx].values, self.get_action(idx))

	def get_action(self, idx):
		wd_low = 2
		wd_high = 2
		for i in range(0,5):
			if self.time.iloc[[idx-i]].index.values[0] == 0 and idx-i>=0:
				wd_low = i
				break
		for i in range(1,5):
			if self.time.iloc[[idx+i]].index.values[0] == 0 and idx+i<__len__():
				wd_high = i-1
				break

		time = self.time.iloc[[idx-wd_low]] - self.time.iloc[[idx+wd_high]]
		rb1_act = [np.linalg.norm((self.rb1.iloc[[idx-wd_low]][['R1X','R1Y']].values - self.rb1.iloc[[idx+wd_high]][['R1X','R1Y']].values) / time),
					np.linalg.norm((self.rb1.iloc[[idx-wd_low]][['R1T']].values - self.rb1.iloc[[idx+wd_high]][['R1T']].values) / time)]

		rb2_act = [np.linalg.norm((self.rb1.iloc[[idx-wd_low]][['R2X','R2Y']].values - self.rb1.iloc[[idx+wd_high]][['R2X','R2Y']].values) / time),
					np.linalg.norm((self.rb1.iloc[[idx-wd_low]][['R2T']].values - self.rb1.iloc[[idx+wd_high]][['R2T']].values) / time)]

		rb3_act = [np.linalg.norm((self.rb1.iloc[[idx-wd_low]][['R3X','R3Y']].values - self.rb1.iloc[[idx+wd_high]][['R3X','R3Y']].values) / time),
					np.linalg.norm((self.rb1.iloc[[idx-wd_low]][['R3T']].values - self.rb1.iloc[[idx+wd_high]][['R3T']].values) / time)]

		return rb1_act+rb2_act+rb3_act

