import torch
import torch.nn as nn
from ..BaseModule import BaseModule


class Model(BaseModule):

	def __init__(self, ent_tot, rel_tot):
		super(Model, self).__init__()
		self.ent_tot = ent_tot
		self.rel_tot = rel_tot

	def forward(self):
		raise NotImplementedError
	
	def predict(self):
		raise NotImplementedError
	

	def get_batch_embeddings(self, data):
		h = self.ent_embeddings(data['batch_h'])
		t = self.ent_embeddings(data['batch_t'])
		r = self.rel_embeddings(data['batch_r'])
		return h, t, r