import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class CDTransE(Model):

	def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None, alpha = 0.5):
		super(CDTransE, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm
		self.alpha = alpha
		# 同一种base embedding下的
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
		self.ent_caus_proj = nn.Linear(self.dim, self.dim)
		self.ent_conf_proj = nn.Linear(self.dim, self.dim)
		self.rel_caus_proj = nn.Linear(self.dim, self.dim)
		self.rel_conf_proj = nn.Linear(self.dim, self.dim)

		self.lp_type = None

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False


	def _calc(self, h, t, r, mode):
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		h_caus = self.ent_caus_proj(h)
		t_caus = self.ent_caus_proj(t)
		r_caus = self.rel_caus_proj(r)
		h_conf = self.ent_conf_proj(h)
		t_conf = self.ent_conf_proj(t)
		r_conf = self.rel_conf_proj(r)
		score_caus = self._calc(h_caus, t_caus, r_caus, mode)
		score_conf = self._calc(h_conf, t_conf, r_conf, mode)
		if self.margin_flag:
			return (self.margin - score_caus, self.margin - score_conf)
		else:
			return (score_caus, score_conf)

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul

	def predict(self, data):
		if self.lp_type == "caus":
			score_caus, score_conf, score_mix = self.forward(data)
			if self.margin_flag:
				score_caus = self.margin - score_caus
				score_conf = self.margin - score_conf
				score = score_caus + self.alpha * score_conf
				return score.cpu().data.numpy()
			else:
				score = score_caus + self.alpha * score_conf
				return score.cpu().data.numpy()
		elif self.lp_type == "mix":
			score_caus, score_conf, score_mix = self.forward(data)
			if self.margin_flag:
				score = self.margin - score_mix
				return score.cpu().data.numpy()
			else:
				score = score_mix
				return score.cpu().data.numpy()