import numpy
import numpy.random


class State(object):
    " Abstract class to define an state. "

    def __hash__(self):
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

        self.reset()

        

    def seed(self, seed):
        numpy.random.seed(seed)

    def reset(self):
        " Reset the state of the machine."
        self.history = [self.start_state]
        self.current_state = self.start_state

    def update_q(self, state, action, correction):
        q = self.get_q(state, action)
        self.q_estimates[(state, action)] = q + self.alpha * correction

    def get_q(self, state, action):

        if (state, action) not in self.q_estimates:
            self.q_estimates[(state, action)] = numpy.random.random()
            
        return self.q_estimates[(state, action)]

    def greedy_action(self, state=None):
        " Compute the E-greedy action. "

        if state is None:
            state = self.current_state

        # List of actions.
        actions = state.actions()

        # List of tuples (action, value)
        actions_value = [(a, self.get_q(state, a)) for a in actions]
        actions_value.sort(key=lambda x: -x[1])

        # We now have the state-actions values in order, add the probability.
        actions_p = [1 - self.epsilon + self.epsilon / len(actions)] + [self.epsilon / len(actions)]

        s = numpy.random.random()

        if s < 1 - self.epsilon:
            return actions_value[0][0]

        # Choose one between the rest.
        s -= 1 - self.epsilon
        s *= (len(actions) - 1) / self.epsilon
        return actions_value[1 + int(s)][0]

    def take_action(self, action):
        next_state, reward = self.current_state.next_state(action)
        return next_state, reward

    def iterate(self):
        " Execute a single episode. "

        self.reset()

        action = self.greedy_action()

        while not self.current_state.is_terminal():

            # Take the greedy action.
            next_state, r = self.take_action(action)

            # Second pair.
            next_action = self.greedy_action(next_state)

            # Update Q(first_state, first_action)
            correction = r + self.gamma * self.get_q(next_state, next_action) - self.get_q(self.current_state, action)
            self.update_q(self.current_state, action, correction)

            self.history.append(next_state)
            self.current_state = next_state
            action = next_action

            
        
