import numpy as np

class TSStruct:
    def __init__(self, num_arm, NoiseScale):
        self.d = num_arm
        self.NoiseScale = NoiseScale

        self.PosteriorMean = np.zeros(self.d)
        self.PosteriorVar = np.ones(self.d)

        self.time = 0

    def updateParameters(self, articlePicked_id, click):
        self.PosteriorMean[articlePicked_id] = self.PosteriorMean[articlePicked_id] + (click - self.PosteriorMean[articlePicked_id]) * (self.PosteriorVar[articlePicked_id] / (self.NoiseScale**2 + self.PosteriorVar[articlePicked_id]))

        self.PosteriorVar[articlePicked_id] = (self.PosteriorVar[articlePicked_id] * self.NoiseScale**2)/(self.PosteriorVar[articlePicked_id] + self.NoiseScale**2)
        self.time += 1

    def getTheta(self):
        return self.PosteriorMean

    def decide(self, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            article_pta = np.random.normal(loc=self.PosteriorMean[article.id], scale=np.sqrt(self.PosteriorVar[article.id]))
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked


class TS:
    def __init__(self, num_arm, NoiseScale):
        self.users = {}
        self.num_arm = num_arm
        self.NoiseScale = NoiseScale
        self.CanEstimateUserPreference = False

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = TSStruct(self.num_arm, self.NoiseScale)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)

    def getTheta(self, userID):
        return self.users[userID].UserArmMean
