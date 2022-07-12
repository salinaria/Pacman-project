# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import numpy as np
import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.values = util.Counter()

    def getQValue(self, state, action):
        return self.values[(state, action)]

    def computeValueFromQValues(self, state):
        legal_actions = self.getLegalActions(state)
        if len(legal_actions) != 0:
            return max([self.getQValue(state, action) for action in legal_actions])
        return 0

    def computeActionFromQValues(self, state):
        legal_actions = self.getLegalActions(state)
        state_max_qvalue = self.computeValueFromQValues(state)
        if len(legal_actions) != 0:
            max_actions = []
            for action in legal_actions:
                if self.getQValue(state, action) == state_max_qvalue:
                    max_actions.append(action)
            return random.choice(max_actions)

        return None

    def getAction(self, state):
        legal_actions = self.getLegalActions(state)
        action = None
        if len(legal_actions) != 0:
            if util.flipCoin(self.epsilon):
                action = random.choice(legal_actions)
            else:
                action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        q_value = self.getQValue(state, action)
        next_qvalue = self.computeValueFromQValues(nextState)
        value = q_value + self.alpha * (reward + self.discount * (next_qvalue) - q_value)
        self.values[(state, action)] = value

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        w = np.array(self.getWeights())
        feature_vector = np.array(self.featExtractor.getFeatures(state, action))
        return np.dot(w, feature_vector)

    def update(self, state, action, nextState, reward):
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        feature_vector = self.featExtractor.getFeatures(state, action)

        for feature in feature_vector:
            self.weights[feature] += self.alpha * difference * feature_vector[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            pass
