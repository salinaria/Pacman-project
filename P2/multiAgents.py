# multiAgents.py
# --------------
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
import math

from util import manhattanDistance
from game import Actions, Directions
import random, util

from game import Agent


def min_distance(current_pos, ghost_state):
    if len(ghost_state) == 0:
        return math.inf
    return min([manhattanDistance(current_pos, gs.getPosition()) for gs in ghost_state])


def min_food_distance(current_pos, food_state):
    mini = math.inf
    for f in food_state.asList():
        if mini > manhattanDistance(current_pos, f):
            mini = manhattanDistance(current_pos, f)
    return mini


def check_scared_time(ScaredTimes):
    for i in ScaredTimes:
        if i < 3:
            return False
    return True


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        current_position = currentGameState.getPacmanPosition()
        score = currentGameState.getScore()
        if currentGameState.isWin() or successorGameState.isWin():
            return math.inf
        if currentGameState.isLose() or successorGameState.isLose():
            return - math.inf

        food_prize = 3
        food_distance = 1
        ghost_distance_prize = 1

        if len(currentGameState.getFood(0).asList()) > len(newFood.asList()):
            score += food_prize

        closest_ghost_distance_current = min_distance(current_position, currentGameState.getGhostStates())
        closest_ghost_distance_next = min_distance(newPos, newGhostStates)

        if closest_ghost_distance_current > closest_ghost_distance_next:
            if check_scared_time(newScaredTimes):
                score += food_prize
        elif closest_ghost_distance_current < closest_ghost_distance_next:
            score += ghost_distance_prize

        if min_food_distance(newPos, newFood) < min_food_distance(current_position, currentGameState.getFood()):
            score += food_distance

        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        return self.minimax(0, gameState, 0)

    def minimax(self, agent_index, state, depth):
        num_agents = state.getNumAgents()

        if state.isWin() or state.isLose() or depth == self.depth or state.getLegalActions(agent_index) == 0:
            return self.evaluationFunction(state)

        if agent_index == 0:
            score = - math.inf
            actions = state.getLegalActions(agent_index)
            best_action = actions[0]
            for action in actions:
                action_score = self.minimax((agent_index + 1) % num_agents,
                                            state.generateSuccessor(agent_index, action), depth)
                if action_score > score:
                    score = action_score
                    best_action = action
            if depth == 0:
                return best_action
            return score

        else:
            score = math.inf
            actions = state.getLegalActions(agent_index)
            for action in actions:

                next_agent = (agent_index + 1) % num_agents

                if next_agent == 0:
                    action_score = self.minimax(next_agent, state.generateSuccessor(agent_index, action), depth + 1)
                else:
                    action_score = self.minimax(next_agent, state.generateSuccessor(agent_index, action), depth)

                if score > action_score:
                    score = action_score
            return score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = - math.inf
        beta = math.inf
        return self.alpha_beta(0, gameState, 0, alpha, beta)

    def alpha_beta(self, agent_index, state, depth, alpha, beta):
        num_agents = state.getNumAgents()

        if state.isWin() or state.isLose() or depth == self.depth or state.getLegalActions(agent_index) == 0:
            return self.evaluationFunction(state)

        if agent_index == 0:
            score = - math.inf
            actions = state.getLegalActions(agent_index)
            best_action = actions[0]
            for action in actions:
                action_score = self.alpha_beta((agent_index + 1) % num_agents,
                                               state.generateSuccessor(agent_index, action), depth, alpha, beta)
                if action_score > score:
                    score = action_score
                    best_action = action
                if action_score > beta:
                    if depth == 0:
                        return best_action
                    return score
                if score > alpha:
                    alpha = score

            if depth == 0:
                return best_action
            return score

        else:
            score = math.inf
            actions = state.getLegalActions(agent_index)
            for action in actions:
                next_agent = (agent_index + 1) % num_agents
                if next_agent == 0:
                    action_score = self.alpha_beta(next_agent, state.generateSuccessor(agent_index, action), depth + 1,
                                                   alpha, beta)
                else:
                    action_score = self.alpha_beta(next_agent, state.generateSuccessor(agent_index, action), depth,
                                                   alpha,
                                                   beta)

                if action_score < score:
                    score = action_score

                if action_score < alpha:
                    return score
                if score < beta:
                    beta = score
            return score


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(0, gameState, 0)

    def expectimax(self, agentIndex, state, depth):
        num_agents = state.getNumAgents()

        if state.isWin() or state.isLose() or depth == self.depth or state.getLegalActions(agentIndex) == 0:
            return self.evaluationFunction(state)

        if agentIndex == 0:
            score = - math.inf
            actions = state.getLegalActions(agentIndex)
            best_action = actions[0]
            for action in actions:
                action_score = self.expectimax((agentIndex + 1) % num_agents,
                                               state.generateSuccessor(agentIndex, action), depth)
                if action_score > score:
                    score = action_score
                    best_action = action
            if depth == 0:
                return best_action
            return score

        else:
            score = 0.0
            actions = state.getLegalActions(agentIndex)
            num_actions = float(len(actions))
            for action in actions:
                next_agent = (agentIndex + 1) % num_agents

                if next_agent == 0:
                    action_score = self.expectimax(next_agent, state.generateSuccessor(agentIndex, action), depth + 1)
                else:
                    action_score = self.expectimax(next_agent, state.generateSuccessor(agentIndex, action), depth)

                score += action_score
            return score / num_actions


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    global max_num_food

    food_prize = 3
    food_distance = 1
    ghost_distance = 2

    score = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    num_food = len(currentGameState.getFood().asList())

    if num_food > max_num_food:
        max_num_food = num_food

    score += food_prize * (max_num_food - num_food)

    if num_food > 0:
        mini_distance = min_food_distance(pos, foods)
        score += food_distance / mini_distance

    closest_ghost_distance_current = min_distance(pos, ghostStates)
    if closest_ghost_distance_current > 0:
        if check_scared_time(scaredTimes):
            score += ghost_distance / closest_ghost_distance_current
        else:
            score -= ghost_distance / closest_ghost_distance_current

    return score


max_num_food = 0

# Abbreviation
better = betterEvaluationFunction
