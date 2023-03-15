import numpy as np

class LinUCBUserStruct:
    def __init__(self, featureDimension, lambda_, delta_, NoiseScale):
        self.d = featureDimension
        self.A = lambda_ * np.identity(n=self.d)
        self.lambda_ = lambda_
        self.delta_ = delta_
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        self.NoiseScale = NoiseScale
        self.UserTheta = np.zeros(self.d)
        self.time = 0
        self.alpha_t = self.NoiseScale * np.sqrt(
            self.d * np.log(1 + self.time / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
            self.lambda_)

    def updateParameters(self, articlePicked_FeatureVector, click):
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b += articlePicked_FeatureVector * click
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.dot(self.AInv, self.b)
        self.time += 1
        self.alpha_t = self.NoiseScale * np.sqrt(
            self.d * np.log(1 + self.time / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
            self.lambda_)

    def getTheta(self):
        return self.UserTheta

    def getA(self):
        return self.A

    def getProb(self, alpha, article_FeatureVector):
        if alpha == -1:
            alpha = self.alpha_t

        mean = np.dot(self.UserTheta, article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv), article_FeatureVector))
        pta = mean + alpha * var
        return pta

class LinUCB:
    def __init__(self, dimension, alpha, lambda_, delta_, NoiseScale):
        self.users = {}
        self.dimension = dimension
        self.alpha = alpha
        self.lambda_ = lambda_
        self.delta_ = delta_
        self.NoiseScale = NoiseScale

        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = LinUCBUserStruct(self.dimension, self.lambda_, self.delta_, self.NoiseScale)
        maxPTA = float('-inf')
        articlePicked = None

        for x in pool_articles:
            x_pta = self.users[userID].getProb(self.alpha, x.featureVector)
            # pick article with highest Prob
            if maxPTA < x_pta:
                articlePicked = x
                maxPTA = x_pta

        return articlePicked, maxPTA

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.featureVector, click)

    def getTheta(self, userID):
        return self.users[userID].UserTheta

    def getAInv(self, userID):
        return self.users[userID].AInv

