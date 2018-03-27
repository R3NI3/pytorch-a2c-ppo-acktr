import pandas as pd
import glob
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import math

def angle_diff(src_angle, target_angle):
	return math.atan2(math.sin(target_angle - src_angle),math.cos(target_angle - src_angle))


class vss_dataset(Dataset):
	def __init__(self, csv_path):
		"""
			Args
				csv_file(string): Path to the csv file
		"""
		filenames = glob.glob(os.path.join(csv_path, "*.csv"))
		df = (pd.read_csv(f, skiprows = 2, dtype = np.float64) for f in filenames)
		self.data = pd.concat(df)
		self.time = self.data['TIMESTAMP_FRAME']
		self.data = self.data.drop('TIMESTAMP_FRAME',1)
		#normalize data
		self.data_norm = (self.data - self.data.mean())/(self.data.max() - self.data.min())

		self.rb1 = self.data[['R1X','R1Y','R1T']]
		self.rb2 = self.data[['R2X','R2Y','R2T']]
		self.rb3 = self.data[['R3X','R3Y','R3T']]

		self.adv1 = self.data[['A1X','A1Y','A1T']]
		self.adv2 = self.data[['A2X','A2Y','A2T']]
		self.adv3 = self.data[['A3X','A3Y','A3T']]

		self.ball = self.data[['BX','BY']]

		self.df_index = self.data.index.values

	def __len__(self):
		return len(self.df_index)

	def __getitem__(self, idx):
		return (torch.FloatTensor(self.data.iloc[idx].values), torch.FloatTensor(self.get_action(idx)))

	def get_action(self, idx):
		wd_low = 4
		wd_high = 0
		for i in range(0,5):
			if idx-i>=0 and self.time.iloc[[idx-i]].index.values[0] == 0:
				wd_low = i
				break
		#for i in range(1,5):
		#	if idx+i<__len__() and self.time.iloc[[idx+i]].index.values[0] == 0:
		#		wd_high = i-1
		#		break

		time = self.time.iloc[[idx+wd_high]].values[0] - self.time.iloc[[idx-wd_low]].values[0]
		time = time/1000.0 # ms -> s
		if (time!=0):
			rb1_act = [np.linalg.norm((self.rb1.iloc[[idx-wd_low]][['R1X','R1Y']].values - self.rb1.iloc[[idx+wd_high]][['R1X','R1Y']].values)) / 100*time,
						angle_diff(self.rb1.iloc[[idx-wd_low]][['R1T']].values, self.rb1.iloc[[idx+wd_high]][['R1T']].values) / time]

			rb2_act = [np.linalg.norm((self.rb2.iloc[[idx-wd_low]][['R2X','R2Y']].values - self.rb2.iloc[[idx+wd_high]][['R2X','R2Y']].values)) / 100*time,
						angle_diff(self.rb2.iloc[[idx-wd_low]][['R2T']].values, self.rb2.iloc[[idx+wd_high]][['R2T']].values) / time]

			rb3_act = [np.linalg.norm((self.rb3.iloc[[idx-wd_low]][['R3X','R3Y']].values - self.rb3.iloc[[idx+wd_high]][['R3X','R3Y']].values)) / 100*time,
						angle_diff(self.rb3.iloc[[idx-wd_low]][['R3T']].values, self.rb3.iloc[[idx+wd_high]][['R3T']].values) / time]

		else:
			return [0.0,0.0,0.0,0.0,0.0,0.0]
		return rb1_act+rb2_act+rb3_act
