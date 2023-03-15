import numpy as np
import math

class LinTSStruct:
	def __init__(self, featureDimension, lambda_, NoiseScale):
		self.d = featureDimension

		self.XTX = np.zeros([self.d, self.d])
		self.XTy = np.zeros(self.d)

		self.Covariance = np.linalg.inv(lambda_ * np.identity(n=self.d) + self.XTX / NoiseScale**2)
		self.Mean = np.dot(self.Covariance, self.XTy / NoiseScale**2)

		self.time = 0
		self.lambda_ = lambda_
		self.NoiseScale = NoiseScale

	def updateParameters(self, articlePicked_FeatureVector, click):
		self.time += 1
		self.XTX += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
		self.XTy += articlePicked_FeatureVector * click
		self.Covariance = np.linalg.inv(self.lambda_ * np.identity(n=self.d) + self.XTX / self.NoiseScale**2)
		self.Mean = np.dot(self.Covariance, self.XTy / self.NoiseScale**2)

	def getSample(self):
		return np.random.multivariate_normal(self.Mean, self.Covariance)

	def getTheta(self):
		return np.dot(np.linalg.inv(self.XTX+self.lambda_*np.identity(self.d)), self.XTy)

class LinTS:
	def __init__(self, dimension, lambda_, NoiseScale):
		self.users = {}
		self.dimension = dimension
		self.lambda_ = lambda_
		self.NoiseScale = NoiseScale
		self.CanEstimateUserPreference = True

	def decide(self, pool_articles, userID):
		if userID not in self.users:
			self.users[userID] = LinTSStruct(self.dimension, self.lambda_, self.NoiseScale)

		maxPTA = float('-inf')
		articlePicked = None

		thetaSample = self.users[userID].getSample()
		for x in pool_articles:
			x_pta = np.dot(thetaSample, x.featureVector)
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta
		return articlePicked, maxPTA

	def updateParameters(self, article_picked, click, userID):
		self.users[userID].updateParameters(article_picked.featureVector, click)

	def getTheta(self, userID):
		return self.users[userID].getTheta()

	def getAInv(self, userID):
		return self.users[userID].Covariance