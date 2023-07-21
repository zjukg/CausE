import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .Model import Model

class CDKGE(Model):

	def __init__(self, model_caus, model_conf, margin, alpha = 0.5, model_type="base", inter_op="add"):
		super(CDKGE, self).__init__(model_caus.ent_tot, model_caus.rel_tot)
		self.model_caus = model_caus
		self.model_conf = model_conf
		self.margin = margin
		self.alpha = alpha
		self.model_type = model_type
		# self.k = k
		self.pred_type = "caus"

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False
		
		self.inter_op = inter_op
		if self.inter_op == "add":
			self.inter_func = self.inter_op_add
		elif self.inter_op == "sub":
			self.inter_func = self.inter_op_sub
		elif self.inter_op == "mult":
			self.inter_func = self.inter_op_mult
		elif self.inter_op == "concat":
			self.inter_func = self.inter_op_concat
		elif self.inter_op == "cross":
			self.inter_func = self.inter_op_cross
		else:
			raise NotImplementedError


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
		score_caus = self.model_caus(data)
		score_conf = self.model_conf(data)
		if self.margin_flag:
			# 此处发生了修改
			return (self.margin - score_caus, self.margin - score_conf)
		else:
			return (score_caus, score_conf)


	def regularization(self, data):
		regul_caus = self.model_caus.regularization(data)
		regul_conf = self.model_conf.regularization(data)
		return (regul_caus + regul_conf) / 2

	
	def l3_regularization(self):
		regul_caus = self.model_caus.l3_regularization()
		regul_conf = self.model_conf.l3_regularization()
		return (regul_caus + regul_conf) / 2


	def predict(self, data):
		if self.pred_type == "caus":
			score_caus = self.model_caus(data)
			if self.model_caus.margin_flag:
				score_caus = self.margin - score_caus
				score = score_caus
				return score.cpu().data.numpy()
			else:
				score = score_caus
				return score.cpu().data.numpy()
		elif self.pred_type == "conf":
			score_conf = self.model_conf(data)
			if self.model_caus.margin_flag:
				score_conf = self.margin - score_conf
				score = score_conf
				return score.cpu().data.numpy()
			else:
				score = score_conf
				return score.cpu().data.numpy()
		elif self.pred_type == "mix":
			score_mix = self.get_batch_mix_score(data)
			if self.model_caus.__class__.__name__ == "DistMult":
				score = -score_mix
				return score.cpu().data.numpy()
			if self.margin_flag:
				score_mix = self.margin - score_mix
				score = score_mix
			else:
				score = score_mix
			return score.cpu().data.numpy()
		elif self.pred_type == "all":
			score_mix = self.get_batch_mix_score(data)
			score_caus = self.model_caus(data)
			if self.model_caus.__class__.__name__ == "DistMult":
				score = -score_mix
				if self.model_caus.margin_flag:
					score_caus = self.margin - score_caus
				score = score + score_caus
				return score.cpu().data.numpy()
			if self.margin_flag:
				score_mix = self.margin - score_mix
			if self.model_caus.margin_flag:
				score_caus = self.margin - score_caus
			score = score_mix + score_caus
			return score.cpu().data.numpy()
		else:
			raise NotImplementedError


	def get_batch_mix_score(self, data):
		if self.model_conf.__class__.__name__ == "ComplEx":
			h1, h2, t1, t2, r1, r2 = self.model_caus.get_batch_embeddings(data)
			h3, h4, t3, t4, r3, r4 = self.model_conf.get_batch_embeddings(data)
			mix_score = self.model_caus._calc(
				self.inter_func(h1, h3),
				self.inter_func(h2, h4),
				self.inter_func(t1, t3),
				self.inter_func(t2, t4),
				self.inter_func(r1, r3),
				self.inter_func(r2, r4)
			)
		else:
			h_caus, t_caus, r_caus = self.model_caus.get_batch_embeddings(data)
			h_conf, t_conf, r_conf = self.model_conf.get_batch_embeddings(data)
			mix_score = self.model_caus._calc(
				self.inter_func(h_caus, h_conf),
				self.inter_func(t_caus, t_conf),
				self.inter_func(r_caus, r_conf),
				data["mode"]
			)
		# CDKGE本身是有margin的
		if self.margin_flag:
			return self.margin - mix_score
		else:
			return mix_score

	def get_batch_similarity_loss(self, data, batch_size):
		h_caus, t_caus, r_caus = self.model_caus.get_batch_embeddings(data)
		h_conf, t_conf, r_conf = self.model_conf.get_batch_embeddings(data)
		h1, t1, r1 = h_caus[:batch_size], t_caus[:batch_size], r_caus[:batch_size]
		h2, t2, r2 = h_conf[:batch_size], t_conf[:batch_size], r_conf[:batch_size]
		h_inter = self.inter_func(h1, h2)
		t_inter = self.inter_func(t1, t2)
		r_inter = self.inter_func(r1, r2)
		h_sim = F.cosine_similarity(h1, h2, dim=1)
		t_sim = F.cosine_similarity(t1, t2, dim=1)
		r_sim = F.cosine_similarity(r1, r2, dim=1)
		h_sim2 = F.cosine_similarity(h1, h_inter, dim=1)
		t_sim2 = F.cosine_similarity(t1, t_inter, dim=1)
		r_sim2 = F.cosine_similarity(r1, r_inter, dim=1)
		sim_sum = (torch.mean(h_sim ** 2) + torch.mean(t_sim ** 2) + torch.mean(r_sim ** 2)) / 3 + (torch.mean(h_sim2 ** 2) + torch.mean(t_sim2 ** 2) + torch.mean(r_sim2 ** 2)) / 3
		return sim_sum
	
	def inter_op_add(self, caus, conf):
		return caus + conf
	
	def inter_op_sub(self, caus, conf):
		return caus - conf
	
	def inter_op_mult(self, caus, conf):
		return caus * conf
	
	def inter_op_concat(self, caus, conf):
		return torch.cat((caus, conf), dim=-1)
	
	def inter_op_cross(self, caus, conf):
		return torch.stack((caus, conf), dim=-1).reshape(caus.shape[0], -1)
