from .Strategy import Strategy
import torch
import torch.nn as nn
import torch.nn.functional as F

class NegativeSampling(Strategy):

	def __init__(self, model = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0, b1 = 0.1, b2 = 0.9):
		super(NegativeSampling, self).__init__()
		self.model = model
		self.loss = loss
		self.batch_size = batch_size
		self.regul_rate = regul_rate
		self.l3_regul_rate = l3_regul_rate
		self.conf_loss = nn.MSELoss(reduction="mean")
		self.beta1 = b1
		self.beta2 = b2
		self.adv_temperature = 1.0

		self.inter_flag = True
		self.loss_w = nn.Parameter(torch.full((5,), fill_value=1.0))
		self.loss_w_record = []

	def _get_positive_score(self, score):
		positive_score = score[:self.batch_size]
		positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score):
		negative_score = score[self.batch_size:]
		negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
		return negative_score
	

	def cal_mse_loss(self, p_score, n_score):
		weights = F.softmax(n_score * self.adv_temperature, dim = -1).detach()
		n_score = (n_score * weights).sum(dim=1)
		return self.conf_loss(p_score.reshape(-1,), n_score.reshape(-1,))

	def forward(self, data):
		score = self.model(data)
		if len(score) == 2:
			score_caus, score_conf = score
			p_caus_score = self._get_positive_score(score_caus)
			n_caus_score = self._get_negative_score(score_caus)
			p_conf_score = self._get_positive_score(score_conf)
			n_conf_score = self._get_negative_score(score_conf)
			loss_caus = self.loss(p_caus_score, n_caus_score)
			loss_conf = self.loss(p_conf_score, n_conf_score)
			if self.inter_flag:
				mix_score = self.model.get_batch_mix_score(data)
				p_mix_score = self._get_positive_score(mix_score)
				n_mix_score = self._get_negative_score(mix_score)
				loss_all = self.loss(p_mix_score, n_mix_score)
				loss_inter = self.loss(p_caus_score, p_mix_score)
				loss_inter2 = self.loss(p_mix_score, p_conf_score)
				# 这三个地位差不多平等
				loss = (loss_caus + loss_conf + loss_all + loss_inter + loss_inter2)
			else:
				raise NotImplementedError
		else:
			# 普通KGE模型
			p_score = self._get_positive_score(score)
			n_score = self._get_negative_score(score)
			loss = self.loss(p_score, n_score)
		if self.regul_rate != 0:
			loss += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss += self.l3_regul_rate * self.model.l3_regularization()
		return loss

		