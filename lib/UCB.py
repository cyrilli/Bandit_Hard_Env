import numpy as np

class UCBStruct:
	def __init__(self, num_arm, NoiseScale):
		self.d = num_arm
		self.NoiseScale = NoiseScale
		self.UserArmMean = np.zeros(self.d)
		self.UserArmTrials = np.zeros(self.d)

		self.time = 0

	def updateParameters(self, articlePicked_id, click):
		self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id] * self.UserArmTrials[articlePicked_id] + click) / (self.UserArmTrials[articlePicked_id]+1)
		self.UserArmTrials[articlePicked_id] += 1
		self.time += 1

	def decide(self, pool_articles):
		for article in pool_articles:
			if self.UserArmTrials[article.id] == 0:
				return article

		max_ucb = float('-inf')
		articlePicked = None
		for article in pool_articles:
			cb = self.NoiseScale * np.sqrt((2 * np.log(self.time)) / float(self.UserArmTrials[article.id]))
			ucb_value = self.UserArmMean[article.id] + cb
			if max_ucb < ucb_value:
				articlePicked = article
				max_ucb = ucb_value

		return articlePicked

	def getTheta(self):
		return self.UserArmMean

class UCB:
	def __init__(self, num_arm, NoiseScale):
		self.users = {}
		self.num_arm = num_arm
		self.NoiseScale = NoiseScale
		self.CanEstimateUserPreference = False

	def decide(self, pool_articles, userID):
		if userID not in self.users:
			self.users[userID] = UCBStruct(self.num_arm, self.NoiseScale)

		return self.users[userID].decide(pool_articles)

	def updateParameters(self, articlePicked, click, userID):

		self.users[userID].updateParameters(articlePicked.id, click)

	def getTheta(self, userID):
		return self.users[userID].getTheta()
