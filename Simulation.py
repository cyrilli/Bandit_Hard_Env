import copy
import numpy as np
from random import sample, shuffle
import datetime
import os.path
import matplotlib.pyplot as plt
import argparse
# local address to save simulated users, simulated articles, and results
from conf import sim_files_folder, save_address
from util_functions import featureUniform, gaussianFeature
from Articles import ArticleManager
from Users import UserManager

from lib.EpsilonGreedyLinearBandit import EpsilonGreedyLinearBandit
from lib.EpsilonGreedyMultiArmedBandit import EpsilonGreedyMultiArmedBandit
from lib.UCB import UCB
from lib.TS import TS
from lib.PHE import PHE
from lib.LinUCB import LinUCB
from lib.LinTS import LinTS
from lib.LinPHE import LinPHE

class simulateOnlineData(object):
	def __init__(self, context_dimension, testing_iterations, plot, article_generator,
				 users, noise=lambda: 0, signature='', NoiseScale=0.0):

		self.simulation_signature = signature

		self.context_dimension = context_dimension
		self.testing_iterations = testing_iterations
		self.batchSize = 1

		self.plot = plot

		self.noise = noise

		self.NoiseScale = NoiseScale
		
		self.article_generator = article_generator
		self.users = users

	def getTheta(self):
		Theta = np.zeros(shape = (self.context_dimension, len(self.users)))
		for i in range(len(self.users)):
			Theta.T[i] = self.users[i].theta
		return Theta
	
	def batchRecord(self, iter_):
		print("Iteration %d"%iter_, " Elapsed time", datetime.datetime.now() - self.startTime)

	def getReward(self, user, pickedArticle):
		return np.dot(user.theta, pickedArticle.featureVector)

	def GetOptimalReward(self, user, articlePool):		
		maxReward = float('-inf')
		maxx = None
		for x in articlePool:	 
			reward = self.getReward(user, x)
			if reward > maxReward:
				maxReward = reward
				maxx = x
		return maxReward, maxx
	
	def getL2Diff(self, x, y):
		return np.linalg.norm(x-y) # L2 norm

	def runAlgorithms(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = self.startTime.strftime('_%m_%d_%H_%M') 
		filenameWriteRegret = os.path.join(save_address, 'AccRegret' + timeRun + self.simulation_signature + '.csv')
		filenameWriteError = os.path.join(save_address, 'AccError' + timeRun + self.simulation_signature + '.csv')
		filenameWritePara = os.path.join(save_address, 'ParameterEstimation' + timeRun + self.simulation_signature + '.csv')

		tim_ = []
		BatchCumlateRegret = {}
		AlgRegret = {}
		BatchCumlateError = {}
		AlgError = {}
		ThetaDiffList = {}
		ThetaDiff = {}
		
		# Initialization
		userSize = len(self.users)
		for alg_name, alg in algorithms.items():
			AlgRegret[alg_name] = []
			AlgError[alg_name] = []
			BatchCumlateRegret[alg_name] = []
			BatchCumlateError[alg_name] = []
			if alg.CanEstimateUserPreference:
				ThetaDiffList[alg_name] = []

		with open(filenameWriteRegret, 'w') as f:
			f.write('Time(Iteration)')
			f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
			f.write('\n')

		with open(filenameWriteError, 'w') as f:
			f.write('Time(Iteration)')
			f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
			f.write('\n')

		with open(filenameWritePara, 'w') as f:
			f.write('Time(Iteration)')
			f.write(','+ ','.join([str(alg_name)+'Theta' for alg_name in ThetaDiffList.keys()]))
			f.write('\n')

		for iter_ in range(self.testing_iterations):
			# prepare to record theta estimation error
			for alg_name, alg in algorithms.items():
				if alg.CanEstimateUserPreference:
					ThetaDiff[alg_name] = 0

			for u in self.users:

				for alg_name, alg in algorithms.items():
					if iter_ == 0:
						articlePool = self.article_generator.simulateArticlePool()
					else:
						articlePool = self.article_generator.simulateArticlePool(iter_/float(self.testing_iterations), u.theta, alg.getTheta(u.id), alg.getAInv(u.id))
					noise = self.noise()
					# get optimal reward for user x at time t
					OptimalReward, OptimalArticle = self.GetOptimalReward(u, articlePool)
					OptimalReward += noise
					pickedArticle, reward_prediction_optimistic = alg.decide(articlePool, u.id)
					reward = self.getReward(u, pickedArticle) + noise
					alg.updateParameters(pickedArticle, reward, u.id)

					regret = OptimalReward - reward  # pseudo regret, since noise is canceled out
					AlgRegret[alg_name].append(regret)
					error = np.abs(reward_prediction_optimistic - self.getReward(u, pickedArticle))  # abs difference between UCB/TS prediction and true mean reward
					AlgError[alg_name].append(error)

					#update parameter estimation record
					if alg.CanEstimateUserPreference:
						ThetaDiff[alg_name] += self.getL2Diff(u.theta, alg.getTheta(u.id))

			for alg_name, alg in algorithms.items():
				if alg.CanEstimateUserPreference:
					ThetaDiffList[alg_name] += [ThetaDiff[alg_name]/userSize]
		
			if iter_%self.batchSize == 0:
				self.batchRecord(iter_)
				tim_.append(iter_)
				for alg_name in algorithms.keys():
					cumRegret = sum(AlgRegret[alg_name])/userSize
					BatchCumlateRegret[alg_name].append(cumRegret)
					cumError = sum(AlgError[alg_name])/userSize
					BatchCumlateError[alg_name].append(cumError)
					print("{0: <16}: cum_regret {1}, cum_error {2}".format(alg_name, cumRegret, cumError))
				with open(filenameWriteRegret, 'a+') as f:
					f.write(str(iter_))
					f.write(',' + ','.join([str(BatchCumlateRegret[alg_name][-1]) for alg_name in algorithms.keys()]))
					f.write('\n')
				with open(filenameWriteError, 'a+') as f:
					f.write(str(iter_))
					f.write(',' + ','.join([str(BatchCumlateError[alg_name][-1]) for alg_name in algorithms.keys()]))
					f.write('\n')
				with open(filenameWritePara, 'a+') as f:
					f.write(str(iter_))
					f.write(','+ ','.join([str(ThetaDiffList[alg_name][-1]) for alg_name in ThetaDiffList.keys()]))
					f.write('\n')

		if (self.plot==True): # only plot
			# plot the results	
			f, axa = plt.subplots(1)
			for alg_name in algorithms.keys():
				axa.plot(tim_, BatchCumlateRegret[alg_name],label = alg_name)
				print('%s: %.2f' % (alg_name, BatchCumlateRegret[alg_name][-1]))
			axa.legend(loc='upper left',prop={'size':9})
			axa.set_xlabel("Iteration")
			axa.set_ylabel("Regret")
			axa.set_title("Accumulated Regret")
			plt.savefig(os.path.join(save_address, "regret" + "_" + str(timeRun) + self.simulation_signature + '.png'), dpi=300,
						bbox_inches='tight', pad_inches=0.0)
			plt.show()

			f, axa = plt.subplots(1)
			for alg_name in algorithms.keys():
				axa.plot(tim_, BatchCumlateError[alg_name],label = alg_name)
				print('%s: %.2f' % (alg_name, BatchCumlateError[alg_name][-1]))
			axa.legend(loc='upper left',prop={'size':9})
			axa.set_xlabel("Iteration")
			axa.set_ylabel("|UCB/TS Prediction - Mean Reward|")
			axa.set_title("Accumulated Error")
			plt.savefig(os.path.join(save_address, "error" + "_" + str(timeRun) + self.simulation_signature + '.png'), dpi=300,
						bbox_inches='tight', pad_inches=0.0)
			plt.show()

			# plot the estimation error of theta
			f, axa = plt.subplots(1)
			time = range(self.testing_iterations)
			for alg_name, alg in algorithms.items():
				if alg.CanEstimateUserPreference:
					axa.plot(time, ThetaDiffList[alg_name], label = alg_name + '_Theta')
	
			axa.legend(loc='upper right',prop={'size':6})
			axa.set_xlabel("Iteration")
			axa.set_ylabel("L2 Diff")
			axa.set_yscale('log')
			axa.set_title("Parameter estimation error")
			plt.savefig(os.path.join(save_address, "estimationError" + "_" + str(timeRun) + self.simulation_signature + '.png'), dpi=300,
						bbox_inches='tight', pad_inches=0.0)
			plt.show()

		finalRegret = {}
		for alg_name in algorithms.keys():
			finalRegret[alg_name] = BatchCumlateRegret[alg_name][:-1]
		return finalRegret

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--contextdim', type=int, default=25, help='Set dimension of context features.')
	parser.add_argument('--actionset', type=str, default='adaptive_adversary_3', help='Set type of context features.')
	parser.add_argument('--namelabel', type=str, default='', help='Set namelabel.')
	args = parser.parse_args()

	## Environment Settings ##
	context_dimension = args.contextdim
	actionset = args.actionset

	# if args.contextdim:
	# 	context_dimension = args.contextdim
	# else:
	# 	context_dimension = 25
	#
	# if args.actionset:
	# 	actionset = args.actionset
	# else:
	# 	actionset = "random"  # "basis_vector" or "random"
	#
	# if args.namelabel:
	# 	name_label = args.namelabel
	# else:
	# 	name_label = ''

	testing_iterations = 2000
	NoiseScale = 0.1  # standard deviation of Gaussian noise
	n_users = 10
	poolArticleSize = 20  # number of articles/arms to present in each round

	name_label = args.namelabel + '_' + 'd'+str(context_dimension) + '_' + 'A'+actionset

	if actionset == "basis_vector":
		poolArticleSize = context_dimension  # there can be at most context_dimension number of basis vectors

	## Set Up Simulation ##
	UM = UserManager(context_dimension, n_users, thetaFunc=gaussianFeature, argv={'l2_limit': 1})
	users = UM.simulateThetafromUsers()
	AM = ArticleManager(dimension=context_dimension, action_set_type=actionset, action_set_size=poolArticleSize, argv={'l2_limit':1})
	# articles = AM.simulateArticlePool(actionset)

	for u in users:
		print(u.theta)

	simExperiment = simulateOnlineData(	context_dimension=context_dimension,
										testing_iterations=testing_iterations,
										plot=True,
										article_generator=AM,
										users = users,
										noise=lambda: np.random.normal(scale=NoiseScale),
										signature=name_label,
										NoiseScale=NoiseScale)

	## Initiate Bandit Algorithms ##
	algorithms = {}

	# algorithms['EpsilonGreedyMultiArmedBandit'] = EpsilonGreedyMultiArmedBandit(num_arm=n_articles, epsilon=None)
	# algorithms['UCB'] = UCB(num_arm=n_articles, NoiseScale=NoiseScale)
	# algorithms['TS'] = TS(num_arm=n_articles, NoiseScale=NoiseScale)
	# algorithms['PHE'] = PHE(num_arm=n_articles, perturbationScale=0.1)

	lambda_ = 0.1
	delta = 1e-1

	# algorithms['EpsilonGreedyLinearBandit'] = EpsilonGreedyLinearBandit(dimension=context_dimension, lambda_=lambda_,epsilon=None)
	algorithms['LinUCB'] = LinUCB(dimension=context_dimension, alpha=-1, lambda_=lambda_, delta_=delta, NoiseScale=NoiseScale)
	algorithms['LinTS'] = LinTS(dimension=context_dimension, NoiseScale=NoiseScale, lambda_=lambda_)
	# algorithms['LinPHE'] = LinPHE(dimension=context_dimension, lambda_=lambda_, perturbationScale=1)

	## Run Simulation ##
	print("Starting for ", simExperiment.simulation_signature)
	simExperiment.runAlgorithms(algorithms)