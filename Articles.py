import numpy as np
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
from random import sample, randint
import json
from numpy import linalg as LA

class Article():	
	def __init__(self, aid, FV=None):
		self.id = aid
		self.featureVector = FV
		

class ArticleManager():
	def __init__(self, dimension, action_set_size, action_set_type, argv):
		self.dimension = dimension
		self.action_set_size = action_set_size
		self.action_set_type = action_set_type
		self.argv = argv

	# def saveArticles(self, Articles, filename, force = False):
	# 	with open(filename, 'w') as f:
	# 		for i in range(len(Articles)):
	# 			f.write(json.dumps((Articles[i].id, Articles[i].featureVector.tolist())) + '\n')
	#
	# def loadArticles(self, filename):
	# 	articles = []
	# 	with open(filename, 'r') as f:
	# 		for line in f:
	# 			aid, featureVector = json.loads(line)
	# 			articles.append(Article(aid, np.array(featureVector)))
	# 	return articles

	def simulateArticlePool(self, time_ratio=None, theta_star=None, theta_hat=None, AInv=None):
		articles = []

		if self.action_set_type == "random":
			# for key in range(self.n_articles):
			# 	featureVector = self.FeatureFunc(self.dimension, argv=self.argv)
			# 	l2_norm = np.linalg.norm(featureVector, ord=2)
			# 	articles.append(Article(key, featureVector/l2_norm))

			feature_matrix = np.empty([self.action_set_size, self.dimension])
			for i in range(self.dimension):
				feature_matrix[:, i] = np.random.normal(0, np.sqrt(1.0*(self.dimension-i)/self.dimension), self.action_set_size)

			for key in range(self.action_set_size):
				featureVector = feature_matrix[key]
				l2_norm = np.linalg.norm(featureVector, ord =2)
				articles.append(Article(key, featureVector/l2_norm ))

		elif self.action_set_type == 'adaptive_adversary_1':
			if time_ratio is not None and theta_star is not None and theta_hat is not None and AInv is not None:
				x_star = theta_star
				articles.append(Article(0, x_star))
				# generate items that orthogonal to x_star
				for key in range(1, self.action_set_size):
					featureVector = find_orth(x_star)
					assert (np.dot(featureVector, x_star) <= 1e-3)
					l2_norm = np.linalg.norm(featureVector, ord =2)
					articles.append(Article(key, featureVector/l2_norm ))
			else:
				feature_matrix = np.empty([self.action_set_size, self.dimension])
				for i in range(self.dimension):
					feature_matrix[:, i] = np.random.normal(0, np.sqrt(1.0 * (self.dimension - i) / self.dimension),
															self.action_set_size)

				for key in range(self.action_set_size):
					featureVector = feature_matrix[key]
					l2_norm = np.linalg.norm(featureVector, ord=2)
					articles.append(Article(key, featureVector / l2_norm))

		elif self.action_set_type == 'adaptive_adversary_2':
			if time_ratio is not None and theta_star is not None and theta_hat is not None and AInv is not None:
				x_star = theta_star
				articles.append(Article(0, x_star))
				# generate items that AInv is least certain about
				vals, vects = LA.eig(AInv)
				# print(vals)
				# print("========")
				# print(vects)
				maxcol = list(vals).index(max(vals))
				featureVector = vects[:, maxcol].reshape(-1,).astype('float64')
				# print(featureVector)
				for key in range(1, self.action_set_size):
					# featureVector = featureVector + np.random.normal(loc=0, scale=0.01, size=self.dimension).reshape(-1,)
					l2_norm = np.linalg.norm(featureVector, ord=2)
					articles.append(Article(key, featureVector / l2_norm))

			else:
				feature_matrix = np.empty([self.action_set_size, self.dimension])
				for i in range(self.dimension):
					feature_matrix[:, i] = np.random.normal(0, np.sqrt(1.0 * (self.dimension - i) / self.dimension),
															self.action_set_size)

				for key in range(self.action_set_size):
					featureVector = feature_matrix[key]
					l2_norm = np.linalg.norm(featureVector, ord=2)
					articles.append(Article(key, featureVector / l2_norm))

		elif self.action_set_type == "basis_vector":
			# This will generate a set of basis vectors to simulate MAB env
			assert self.action_set_size == self.dimension
			feature_matrix = np.identity(self.action_set_size)
			for key in range(self.action_set_size):
				featureVector = feature_matrix[key]
				articles.append(Article(key, featureVector))

		return articles


from numpy.linalg import lstsq
from scipy.linalg import orth
def find_orth(O):
	# adapted from https://stackoverflow.com/questions/50660389/generate-a-vector-that-is-orthogonal-to-a-set-of-other-vectors-in-any-dimension
	O = O.reshape(O.shape[0], 1)
	rand_vec = np.random.rand(O.shape[0], 1)
	A = np.hstack((O, rand_vec))
	b = np.zeros(O.shape[1] + 1)
	b[-1] = 1
	return lstsq(A.T, b)[0]

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
def projection_in_norm(x, M):
	"""Projection of x to simplex indiced by matrix M. Uses quadratic programming."""
	x = np.mat(x).T
	M = np.mat(M)
	m = M.shape[0]

	P = matrix(2 * M)
	q = matrix(-2 * M * x)
	G = matrix(-np.eye(m))
	h = matrix(np.zeros((m, 1)))
	A = matrix(np.ones((1, m)))
	b = matrix(1.0)
	sol = solvers.qp(P, q, G, h, A, b)
	return np.squeeze(sol["x"])
