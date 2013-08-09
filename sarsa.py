import numpy
import numpy.random
from multiprocessing import Queue


class State(object):
    " Abstract class to define an state. "

    def __hash__(self):
        raise NotImplementedError("__hash__ function must be implemented.")

    @staticmethod
    def getfromhash(hash):
        raise NotImplementedError("__hash__ function must be implemented.")

    def __eq__(self, other):
        raise NotImplementedError("__eq__ function must be implemented.")

    def next_state(self, state):
        raise NotImplementedError("A tuple (new state, reward) must be returned.")

    def is_terminal(self):
        raise NotImplementedError("Check if the current state is terminal.")

    def actions(self):
        raise NotImplementedError("List of actions in the state.")

    
class Sarsa(object):

    def __init__(self, start_state, epsilon=0.1, alpha=0.1, gamma=0.8):
        self.start_state = start_state

        # Keep the estimates for Q(s, a)
        self.q_estimates = {}

        # Epsilon for the E-greedy policy.
        self.epsilon = epsilon

        # Alpha/
        self.alpha = alpha

        # Lambda
        self.gamma = gamma
        
    def seed(self, seed):
        numpy.random.seed(seed)

    def update_q(self, state, action, correction):
        q = self.get_q(state, action)
        self.q_estimates[(state, action)] = q + self.alpha * correction

    def get_q(self, state, action):

        if (state, action) not in self.q_estimates:
            self.q_estimates[(state, action)] = numpy.random.random()
            
        return self.q_estimates[(state, action)]

    def greedy_action(self, state, epsilon=None):
        " Compute the E-greedy action. "

        if epsilon is None:
            epsilon = self.epsilon

        # List of actions.
        actions = state.actions()

        # List of tuples (action, value)
        actions_value = [(a, self.get_q(state, a)) for a in actions]
        actions_value.sort(key=lambda x: -x[1])

        # We now have the state-actions values in order, add the probability.
        actions_p = [1 - epsilon + epsilon / len(actions)] + [epsilon / len(actions)]

        s = numpy.random.random()

        if s < 1 - epsilon:
            return actions_value[0][0]

        # Choose one between the rest.
        s -= 1 - epsilon
        s *= (len(actions) - 1) / epsilon
        return actions_value[1 + int(s)][0]

    def iterate(self):
        " Execute a single episode. "

        state = self.start_state
        action = self.greedy_action(state)

        # Save the history.
        history = [state]
        corrections = []

        while not state.is_terminal():

            # Take the greedy action.
            next_state, r = state.next_state(action)

            # Second pair.
            next_action = self.greedy_action(next_state)

            # Update Q(first_state, first_action)
            correction = r + self.gamma * self.get_q(next_state, next_action) - self.get_q(state, action)
            self.update_q(state, action, correction)

            # Save history.
            corrections.append(correction)
            history.append(next_state)

            # Update state.
            state = next_state
            action = next_action

        return history, corrections

            
    def eval(self, max_iter=100):
        """ Just evaluate the current policy."""

        # Save the history in different array.
        history = [self.start_state]
        state = self.start_state
        reward = 0

        for it in range(max_iter):

            # Complete greedy action, no exploration!.
            action = self.greedy_action(state=state, epsilon=0)

            # Take the greedy action.
            state, r = state.next_state(action)
            reward += r

            # Save the state.
            history.append(state)

            if state.is_terminal():
                break

        return history, reward


class SarsaTraces(Sarsa):

    def __init__(self, start_state, epsilon=0.1, alpha=0.1, gamma=0.8, Lambda=0.1):

        self.start_state = start_state

        # Keep the estimates for Q(s, a)
        self.q_estimates = {}

        # Keep the traces
        self.traces = {}

        # Epsilon for the E-greedy policy.
        self.epsilon = epsilon

        # Alpha/
        self.alpha = alpha

        # Gamma
        self.gamma = gamma

        # Lambda
        self.Lambda = Lambda

        self.reset()

    def reset(self):
        " Reset the state of the machine."
        self.history = [self.start_state]
        self.corrections = []
   
        # This is not part of the algorithm but I don't see the point in keeping this on
        # different episodes, and cleaning this will increase the performance.
        self.traces = {}

    def update_trace(self, state, action):
        # Update and get acts as an sparsed matrix based on key-value.
        # Check on:
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.html
        self.traces[(state, action)] = self.gamma * self.Lambda * self.get_trace(state, action)

    def get_trace(self, state, action):
        if (state, action) not in self.traces:
            self.traces[(state, action)] = 0
        return self.traces[(state, action)]

    def update_iter(self, state, action, correction):
        self.update_q(state, action, correction)
        self.update_trace(state, action)        

    def iterate(self):
        " Execute a single episode. "

        self.reset()
        
        state = self.start_state
        action = self.greedy_action(state)

        while not state.is_terminal():

            # Take the greedy action.
            next_state, r = state.next_state(action)

            # Second pair.
            next_action = self.greedy_action(next_state)

            # Update Q(first_state, first_action)
            correction = r + self.gamma * self.get_q(next_state, next_action) - self.get_q(state, action)
            self.traces[(state, action)] = self.get_trace(state, action) + 1
            
            # Update just the states with non-zero trace.
            for key, value in self.traces.items():
                self.update_q(key[0], key[1], correction * value)
                self.update_trace(*key)

            self.history.append(next_state)
            state = next_state
            action = next_action
        


class SarsaPartition(Sarsa):
    """
    Sarsa with partition
    ====================

    I will try to implement a patitioned sarsa based on ideas of the paper:
    http://www.ejournal.org.cn/Jweb_cje/EN/article/downloadArticleFile.do?attachType=PDF&id=7638
    The basic idea is that we can define a partition in the space state with some nice properties that I will try to express nicely if this works.

    I will try to implement a way to split the big problem in little problems adding a method
    *part* in the state class which says to which part the state belongs, if we hit a
    `next_state` in a different class we are going to need to add a problem starting on
    that state to the scheduler, and mark it to update, then any other subproblem that hit
    this one will wait for the update mark it again and add a new subproblem starting in the
    same point.

    We need to add a mark to the Q values, this mark will control that a value of Q in
    the boundary of a partition will be used just once by a subsolver.

    
    """

    def __init__(self, *args, **kwargs):
        super(SarsaPartition, self).__init__(*args, **kwargs)

        # We're going to store the states in this queues.
        self.queues = {}

        # We need to add the queues of the parts.
        for p in kwargs['parts']:
            self.queues[p] = Queue()

    def iterate(self, part):
        " Execute a single episode. "

        # Fetch a problem from part 
        history, corrections, state = self.queue[part].get()

        self.reset()
        
        state = self.start_state
        action = self.greedy_action(state)

        while not state.is_terminal():

            # Take the greedy action.
            next_state, r = state.next_state(action)

            # Second pair.
            next_action = self.greedy_action(next_state)

            # Update Q(first_state, first_action)
            correction = r + self.gamma * self.get_q(next_state, next_action) - self.get_q(state, action)
            self.traces[(state, action)] = self.get_trace(state, action) + 1
            
            # Update just the states with non-zero trace.
            for key, value in self.traces.items():
                self.update_q(key[0], key[1], correction * value)
                self.update_trace(*key)

            self.history.append(next_state)
            state = next_state
            action = next_action
