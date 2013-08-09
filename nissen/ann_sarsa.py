from UserDict import UserDict
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer, LinearLayer
from random import random
import sys
import argparse
import json


class EligibilityTraces(UserDict):
    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = 0
        return self.data[key]    


class StateActionCache(UserDict):
    """
    This class will implement the cache described in
    section 5.4 in Nissen Thesis [1].
    [1] http://leenissen.dk/rl/Steffen_Nissen_Thesis2007_Print.pdf
    """

    def __init__(self, input_size, hidden_size=10, cache_size=1000):
        self.data = {}
        self.cache_size = cache_size
        self.fifo = []

        # Create the network in here.
        self.net = buildNetwork(input_size, hidden_size, 1, outclass=LinearLayer, bias=True)

    def __getitem__(self, key):
        if key not in self.data:            
            self.data[key] = (1, self.net.activate(key[0]._to_ann(key[1]))[0])
        return self.data[key][1]

    def __delitem__(self, key):

        # Get current number of appearances,
        n, v = self.data.get(key, (0, None))

        if n <= 1:
            del self.data[key]
        else:
            self.data[key] = (n - 1, v)
        
        
    def __setitem__(self, key, value):

        if self.filled():
            del self[self.fifo.pop(0)]

        n = 1
        if key in self.data:
            n = self.data[key][0] + 1            

        self.data[key] = (n, value)
        self.fifo.append(key)
        
    def filled(self):
        return len(self.fifo) > self.cache_size

    def train(self):
        " Train the network. "
        
        ds = SupervisedDataSet(self.net.indim, self.net.outdim)
        n = len(self.fifo)
        for k, v in self.data.items():
            d = (k[0]._to_ann(k[1]), [v[1]])
            for i in range(v[0]):
                ds.addSample(*d)

        ret = None
        if len(ds) != 0:
            trainer = BackpropTrainer(self.net, ds)
            ret = trainer.train()
            
        # Clean the inner data.
        self.data = {}
        self.fifo = []
        
        return ret

class QSarsa(object):

    def __init__(self, state, cache_size=20000, hidden_size=8, Epsilon=0.8, Alpha=0.1, Gamma=0.8, Tao=0.1, Sigma=0.5, Lambda=0.3):

        # Starting state.
        self.initial_state = state

        # Save the parameters.
        self.Epsilon = Epsilon
        self.Gamma = Gamma
        self.Alpha = Alpha
        self.Tao = Tao
        self.Sigma = Sigma
        self.Lambda = Lambda

        # Q Cache.
        self.Q = StateActionCache(state._to_ann_size(), cache_size=cache_size, hidden_size=hidden_size)

        # Eligibility traces.
        self.e = EligibilityTraces()

    def select_action(self, state):
        " Select an action on the state."

        # List of tuples (action, value)
        actions = [(a, self.Q[(state, a)]) for a in state.actions]
        actions.sort(key=lambda x: -x[1])

        # Check if we're greed in this case.
        s = random()
        if s <= 1 - self.Epsilon:
            return actions[0]

        # Choose one between the rest.
        s -= 1 - self.Epsilon
        s *= (len(actions) - 1) / self.Epsilon
        return actions[1 + int(s)][0], actions[0][1]

    def select_best_action(self, state):
        " Select an action on the state."

        # List of tuples (action, value)
        actions = [(a, self.Q[(state, a)]) for a in state.actions]
        actions.sort(key=lambda x: -x[1])

        # Check if we're greed in this case.
        return actions[0]

    def evaluate(self):
        " Evaluate the system."

        state = self.initial_state
        it = 0

        history = [state]
        reward_total = 0

        while not state.is_terminal and it < self.initial_state.size ** 2:
            action, best_reward = self.select_best_action(state)
            state, reward = state.next_state(action)
            reward_total += reward
            history.append(state)
            it += 1

        return reward_total, history

    def run(self, it):
        " Run a episode."

        # Clear the traces.
        self.e = EligibilityTraces()

        # Starting state.
        state = self.initial_state

        # Select action.
        action, reward = self.select_action(state)

        while not state.is_terminal:

            # print len(self.Q.fifo), state.position, len(state.food)

            # Iterate.
            state, action = self.iterate(state, action)

            if self.Q.filled():

                # Train the neural network,
                print json.dumps({'iteration': it, 'train': self.Q.train()})                       
        print json.dumps({'iteration': it, 'train': self.Q.train()})       
#        self.evaluate()

    def iterate(self, state, action):
        " Do a single iteration from pair to pair."

        # Take the greedy action.
        next_state, reward = state.next_state(action)

        # Second action.
        next_action, max_reward = self.select_action(next_state)

        # compute the correction.
        delta = reward + self.Gamma *((1 - self.Sigma) * max_reward + self.Sigma * self.Q[(next_state, next_action)]) - self.Q[(state, action)]

        # Update eligibility.
        self.e[(state, action)] += 1

        # # Update the Q and e.
        for s, a in self.e.data.keys():
             if self.e[(s, a)] < (self.Gamma * self.Lambda) ** 20:
                 self.e.data.pop((s, a))
             else:
                 self.Q[(s, a)] += self.e[(s, a)] * delta * self.Lambda * self.Alpha
                 self.e[(s, a)] *= self.Gamma * self.Lambda
             
        # Return the next state and actions.
        return next_state, next_action


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description='Combine Sarsa with Neural Networks.')
    parser.add_argument('--size', type=int, dest="size", help='Size of the barn.', default=10)
    parser.add_argument('--cache-size', type=int, dest="cache_size", help='Size of the cache.', default=1000)
    parser.add_argument('--hidden-size', type=int, dest="hidden_size", help='Size of the hidden layer of neurons.', default=25)
    parser.add_argument('--epsilon', type=float, dest="Epsilon", help="Epsilo value, close to one is complete random, 0 is greedy", default=0.6)
    parser.add_argument('--alpha', type=float, dest="Alpha", help="Step paramenter, 1 is fast close to zero is slow but secure.", default=0.1)
    parser.add_argument('--gamma', type=float, dest="Gamma", help="0 is no memory, 1 is funes.", default=0.2)    
    parser.add_argument('--lambda', type=float, dest="Lambda", help="Control the eligibility traces.", default=0.2)    
    parser.add_argument('--iter-step', type=int, dest="iter_step", help="Evaluate and print each number of steps.", default=100)    
    parser.add_argument('--iter-total', type=int, dest="iter_step", help="Stop at this steps", default=50000)    
    args = parser.parse_args()

    # Intial state.
    from state import BarnState
    initial_state = BarnState.initial_state(args.size)

    # Print to stdout.
    print json.dumps(vars(args))
    print json.dumps({"initial": {"food": initial_state.food, "position": initial_state.position}})

    # Create a single sarsa.
    sarsa = QSarsa(initial_state, cache_size=args.cache_size, hidden_size=args.hidden_size, Epsilon=args.Epsilon, Alpha=args.Alpha,
                   Gamma=args.Gamma, Lambda=args.Lambda, Sigma=1)
    for i in range(100000):
        sarsa.run(i)
        if i % args.iter_step == 0:
            reward, history = sarsa.evaluate()
            # print reward_total, [(s.position, len(s.food)) for s in history]
            print json.dumps({"iteration": i, "eval": {"reward": reward, "history": [(s.position, len(s.food)) for s in history]}})
