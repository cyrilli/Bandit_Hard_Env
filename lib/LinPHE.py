import numpy as np
import math

class LinPHEStruct:
	def __init__(self, featureDimension, lambda_, a):
		self.d = featureDimension
		self.lambda_ = lambda_
		self.time = 0

		self.armFeatureVecs = {}
		self.armTrials = {}
		self.armCumReward = {}
		self.a = a
		# if a != -1:
		# 	self.a = a
		# else:
		# 	# see Perturbed-History Exploration in Stochastic Linear Bandits Table 1
		# 	c1 = 0.5 * np.sqrt(
		# 		self.d * np.log(testing_iterations + testing_iterations**2 / (self.d * self.lambda_))) + np.sqrt(self.lambda_)
		# 	self.a = math.ceil(16 * c1**2)
		print("self.a {}".format(self.a))
		self.f_noiseless = np.zeros(self.d)
		self.B = np.zeros((self.d, self.d))
		self.UserTheta = np.zeros(self.d)

		self.G_0 = lambda_ * (self.a + 1) * np.identity(self.d)

	def updateParameters(self, article_picked, click):
		self.time += 1
		if article_picked.id not in self.armTrials:
			self.armTrials[article_picked.id] = 0
			self.armFeatureVecs[article_picked.id] = article_picked.featureVector
			self.armCumReward[article_picked.id] = 0
		self.armTrials[article_picked.id] += 1
		self.armCumReward[article_picked.id] += click

		self.B += np.outer(article_picked.featureVector, article_picked.featureVector)
		G = (self.a + 1) * self.B + self.G_0

		perturbed_f = np.zeros(self.d)
		for armID, armCumReward in self.armCumReward.items():
			perturbed_f += self.armFeatureVecs[armID] * (armCumReward + np.random.binomial(self.a*self.armTrials[armID], 0.5))

		self.f_noiseless += article_picked.featureVector * click
		self.UserTheta = np.dot(np.linalg.inv(G), perturbed_f)

	def getProb(self, article_featureVector):
		return np.dot(self.UserTheta, article_featureVector)

	def getTheta(self):
		return np.dot(np.linalg.inv(self.B+self.lambda_*np.identity(self.d)), self.f_noiseless)

class LinPHE:
	def __init__(self, dimension, lambda_, perturbationScale=1):
		self.dimension = dimension
		self.perturbationScale = perturbationScale
		self.lambda_ = lambda_

		self.users = {}

		self.CanEstimateUserPreference = True

	def decide(self, pool_articles, userID):
		if userID not in self.users:
			self.users[userID] = LinPHEStruct(self.dimension, self.lambda_, self.perturbationScale)

		maxPTA = float('-inf')
		articlePicked = None


		for x in pool_articles:
			if x.id not in self.users[userID].armTrials:
				return x
			x_pta = self.users[userID].getProb(x.featureVector)
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta
		return articlePicked

	def updateParameters(self, article_picked, click, userID):
		self.users[userID].updateParameters(article_picked, click)

	def getTheta(self, userID):
		return self.users[userID].getTheta()