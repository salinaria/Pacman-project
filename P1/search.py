# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    from util import Stack
    stack = Stack()
    visited_states = []
    path = []
    if problem.isGoalState(problem.getStartState()):
        return []
    else:
        stack.push((problem.getStartState(), []))
        while not stack.isEmpty():
            current_state, path = stack.pop()
            visited_states.append(current_state)
            if problem.isGoalState(current_state):
                return path
            else:
                for successor in problem.getSuccessors(current_state):
                    if successor[0] not in visited_states:
                        new_path = path + [successor[1]]
                        stack.push((successor[0], new_path))
        if stack.isEmpty():
            return stack


def breadthFirstSearch(problem):
    from util import Queue
    queue = Queue()
    visited_states = []
    path = []
    if problem.isGoalState(problem.getStartState()):
        return []
    else:
        queue.push((problem.getStartState(), []))
        while not queue.isEmpty():
            current_state, path = queue.pop()
            visited_states.append(current_state)
            if problem.isGoalState(current_state):
                return path
            else:
                for successor in problem.getSuccessors(current_state):
                    if successor[0] not in visited_states:
                        flag = True
                        for i in queue.list:
                            if successor[0] == i[0]:
                                flag = False
                        if flag:
                            new_path = path + [successor[1]]
                            queue.push((successor[0], new_path))
        if queue.isEmpty():
            return queue


def uniformCostSearch(problem):
    from util import PriorityQueue
    queue = PriorityQueue()
    visited_states = []
    path = []
    if problem.isGoalState(problem.getStartState()):
        return []
    else:
        queue.push((problem.getStartState(), []), 0)
        while not queue.isEmpty():
            if queue.isEmpty():
                return []
            current_state, path = queue.pop()
            if problem.isGoalState(current_state):
                return path
            if current_state not in visited_states:
                visited_states.append(current_state)
                for successor in problem.getSuccessors(current_state):
                    if successor[0] not in visited_states:
                        new_path = path + [successor[1]]
                        queue.update((successor[0], new_path), problem.getCostOfActions(new_path))
        if queue.isEmpty():
            return queue


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    from util import PriorityQueue
    queue = PriorityQueue()
    visited_states = []
    if problem.isGoalState(problem.getStartState()):
        return []
    queue.push((problem.getStartState(), [], 0), 0)
    while True:
        if queue.isEmpty():
            return []
        current_state, path, current_cost = queue.pop()
        if problem.isGoalState(current_state):
            return path
        if current_state not in visited_states:
            visited_states.append(current_state)
            for successor in problem.getSuccessors(current_state):
                if successor[0] not in visited_states:
                    cost = current_cost + successor[2]
                    total_cost = cost + heuristic(successor[0], problem)
                    queue.push((successor[0], path + [successor[1]], cost), total_cost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
