import numpy as np

class PHEStruct:
    def __init__(self, num_arm, perturbationScale):
        self.d = num_arm
        self.perturbationScale = perturbationScale
        self.UserArmMean = np.zeros(self.d)
        self.Ti = np.zeros(self.d)
        self.Vi = np.zeros(self.d)
        self.time = 0

    def updateParameters(self, articlePicked_id, click):
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id]*self.Ti[articlePicked_id] + click) / (self.Ti[articlePicked_id]+1)
        self.Ti[articlePicked_id] += 1
        self.Vi[articlePicked_id] += click
        self.time += 1

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            if self.Ti[article.id] > 0:
                s = self.Ti[article.id]
                U = np.sum(np.random.binomial(int(self.perturbationScale*s), 0.5))
                article_pta = (self.Vi[article.id] + U) / ((self.perturbationScale+1)*s)
            else:
                article_pta = float('inf')

            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked

class PHE:
    def __init__(self, num_arm, perturbationScale=0.1):
        self.users = {}
        self.num_arm = num_arm
        self.perturbationScale = perturbationScale
        self.CanEstimateUserPreference = False

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = PHEStruct(self.num_arm,self.perturbationScale)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)

    def getTheta(self, userID):
        return self.users[userID].UserArmMean
