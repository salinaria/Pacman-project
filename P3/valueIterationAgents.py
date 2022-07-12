# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for i in range(self.iterations):
            iteration_values = util.Counter()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    actions = self.mdp.getPossibleActions(state)
                    max_value = max([self.computeQValueFromValues(state, action) for action in actions])
                    iteration_values[state] = max_value
                else:
                    self.values[state] = self.mdp.getReward(state, 'exit', '')
            self.values = iteration_values

    def getValue(self, state):
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        transition = self.mdp.getTransitionStatesAndProbs(state, action)
        value = 0
        for next, t in transition:
            transition_reward = self.mdp.getReward(state, action, next)
            value += t * (transition_reward + self.discount * self.values[next])

        return value

    def computeActionFromValues(self, state):
        state_action = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            state_action[action] = self.computeQValueFromValues(state, action)
        return state_action.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    def __init__(self, mdp, discount=0.9, iterations=1000):
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        for i in range(self.iterations):
            if not self.mdp.isTerminal(states[i % len(states)]):
                actions = self.mdp.getPossibleActions(states[i % len(states)])
                max_value = max([self.computeQValueFromValues(states[i % len(states)], action) for action in actions])
                self.values[states[i % len(states)]] = max_value
            else:
                self.values[states[i % len(states)]] = self.mdp.getReward(states[i % len(states)], 'exit', '')


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        priority_queue = util.PriorityQueue()
        moves = ['north', 'south', 'east', 'west']
        predecessors = {}
        states = self.mdp.getStates()

        for state in states:
            self.values[state] = 0
            pred = set()
            if not self.mdp.isTerminal(state):
                for state_2 in states:
                    if not self.mdp.isTerminal(state_2):
                        for move in moves:
                            if move in self.mdp.getPossibleActions(state_2):
                                for next_state, t in self.mdp.getTransitionStatesAndProbs(state_2, move):
                                    if (next_state == state) and (t > 0):
                                        pred.add(state_2)
            predecessors[state] = pred

        for state in states:
            if not self.mdp.isTerminal(state):
                value = self.values[state]
                max_value = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
                diff = abs(value - max_value)
                priority_queue.push(state, -diff)

        for i in range(self.iterations):
            if priority_queue.isEmpty():
                return
            state = priority_queue.pop()
            self.values[state] = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
            for pred in predecessors[state]:
                max_ = max([self.getQValue(pred, action) for action in self.mdp.getPossibleActions(pred)])
                diff = abs(self.values[pred] - max_)
                if diff > self.theta:
                    priority_queue.update(pred, -diff)
