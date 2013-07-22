import numpy
import numpy.random
from random import randint, choice
from UserDict import UserDict
from multiprocessing import Process, Manager, Pool
from Queue import Queue
import pylab


class StateActionValues(UserDict):
    """
    We should handle the blocking in here, if we know that there is not going to be any
    problem (interior of the set) we can use the normal dict api, otherwise, the get and set
    methods are the one that checks for blocks.
    """

    def __getitem__(self, key):
        if key not in self.data:
            #self.data[key] = numpy.random.random()
            self.data[key] = 0
        return self.data[key]    


class SarsaAgent(Process):
    """
    This object will run the iterative process needed in sarsa.
    
    """

    def __init__(self, part, inbox, outbox, max_iter=20, Epsilon=0.8, Alpha=0.1, Gamma=0.8):
        Process.__init__(self)

        # Save the parameters.
        self.Epsilon = Epsilon
        self.Alpha = Alpha
        self.Gamma = Gamma

        # Keep track of all the parts.
        self.part = part

        # We receive jobs in this queue.
        self.inbox = inbox

        # We sent stopping states in this box.
        self.outbox = outbox

        # Max number of iterations.
        self.max_iter = max_iter

    def run(self):

        while True:

            # Get a job from the queue.
            job = self.inbox.get()

            # Stop condition.
            if job is None:
                break

            # Job is just a pair state-action to start.
            state, action, own_Q, extra_Q = job

            # Recover the Q values.
            self.own_Q = StateActionValues(own_Q)
            self.extra_Q = StateActionValues(extra_Q)

            # Compute greedy action
            if action is None:
                action = self.greedy_action(state)

            # Iterate a bunch of times with this condition.
            res = []
            for it in range(self.max_iter):                
                res.append(self.iterate(state, action))
                
            # A message from the agent to the scheduler.
            self.outbox.put({'part': self.part, 'extra_Q': self.extra_Q.data, 'own_Q': self.own_Q.data, 'results': res})

    def greedy_action(self, state, Epsilon=None):
        " Compute the E-greedy action. "

        # Get the current epsilon or the stored one (to control the ??greedyness??)
        Epsilon = Epsilon or self.Epsilon 

        # List of actions.
        actions = state.actions()

        # List of tuples (action, value)
        if state.part == self.part:
            actions_value = [(a, self.own_Q[(state, a)]) for a in actions]
        else:
            actions_value = [(a, self.extra_Q[(state, a)]) for a in actions]
            
        actions_value.sort(key=lambda x: -x[1])

        # We now have the state-actions values in order, add the probability.
        actions_p = [1 - Epsilon + Epsilon / len(actions)] + [Epsilon / len(actions)]

        # Check if we're greed in this case.
        s = numpy.random.random()
        if s < 1 - Epsilon:
            return actions_value[0][0]

        # Choose one between the rest.
        s -= 1 - Epsilon
        s *= (len(actions) - 1) / Epsilon
        return actions_value[1 + int(s)][0]

    def iterate(self, initial_state, initial_action):

        state = initial_state
        action = initial_action

        while (not state.is_terminal()) and (state.part == self.part):

            # Take the greedy action.
            next_state, reward = state.next_state(action)
            
            # Second action.
            next_action = self.greedy_action(next_state)
            
            # If next state is in different part the reward is the estimated value.
            # We still use the value in this part, since is job of the scheduler to
            # update the value with the possible result in the other part.
            if next_state.part != self.part:
                reward = self.extra_Q[(next_state, next_action)]
                next_q = reward
            else:
                next_q = self.own_Q[(next_state, next_action)]
            
            # Update Q(first_state, first_action)
            correction = reward + self.Gamma * next_q - self.own_Q[(state, action)]
            self.own_Q[(state, action)] += self.Alpha * correction
            
            # Update state.
            state = next_state
            action = next_action

        
        return state, action

                

class Sarsa(object):
    """
    In the state space we can define some nice concept:

     - Neighbor of a state x: all the state reacheble from the state after an action.
     - Interior of X: the set of states with neigbhors in X.
     - Closure of X: the union of neigbhors of the state x for all x in X.
     - Boundary of X: Closure of X - Interior of X.
     
    If we let sarsa to act on the interior of a set X we have no problems, we need to check
    if an state x in X is in the boundary of X.

    A subproblem starts when is fetched from the queue of subproblems, there must be a
    master process checking to add more starting problems which is just the starting state.

    Once the iterative algorithm hit a state in the boundary we check if next_state is outside
    the set, in that case we check if there is a block in the Q(next_state, next_action) and
    we wait for it, otherwise we get the value and put a block, in both cases the next step
    is to add a subproblem starting in (next_state, next_action).

    We need to ensure that the number of starting subproblems is considerable big that the
    max of the size of the boundaries.

    """

    def __init__(self, manager, initial_state, Epsilon=0.8, Alpha=0.1, Gamma=0.8):

        # Save the parameters.
        self.Epsilon = Epsilon
        self.Alpha = Alpha
        self.Gamma = Gamma

        # Keep track of all the parts.
        self.parts = initial_state.parts

        # Create the queues for the passing jobs.
        self.outbox = dict((p, manager.Queue()) for p in self.parts)
        self.inbox = manager.Queue()

        # Keep the global state-action values in here.
        self.Q = dict((p, StateActionValues()) for p in self.parts)
        
        # Create the processes for each part.
        self.agents = dict((p, SarsaAgent(p, self.outbox[p], self.inbox, Epsilon=Epsilon, Gamma=Gamma, Alpha=Alpha)) for p in self.parts)

        # Each part has an unknown set of states in other parts.
        self.needed_Q = dict((p, set([])) for p in self.parts)

        # Initial states.
        self.initial_states = []

        # Save the initial state.
        self.initial_state = initial_state

        # Start all the agents.
        [self.agents[a].start() for a in self.agents]


    def seed(self, seed):
        numpy.random.seed(seed)


    def select_jobs(self, n):
        " Select n jobs to be scheduled."
        return set([choice(self.initial_states) for i in range(n)])

        return sorted(self.initial_states, key=lambda x: x[0].part)[:n]

    def iterate(self, max_iter=100):
        " Execute a single episode. "

        # Start the iterations counter, iterations are counted each time we reacheble
        # a global stopping state.
        it = 0

        # Start by sending an initial state.        
        self.outbox[self.initial_state.part].put((self.initial_state, None, {}, {}))

        # Running now
        running = 1

        while it < max_iter:

            if len(self.initial_states) > 0:

                # Get the jobs to send
                # jobs = set([choice(self.initial_states) for i in range(10 - running)])
                jobs = self.select_jobs(max(4 - running, 0))
                
                for job in jobs:
                    
                    state, action = job
                    part = state.part

                    # Get own and extra Q values.
                    own_Q = self.Q[part].data
                    extra_Q = dict((k, self.Q[k[0].part][k]) for k in self.needed_Q[part])

                    # Submit job with Q values.
                    self.outbox[part].put((state, action, own_Q, extra_Q))

                    # One more is running.
                    running += 1
            
            # Read a message.
            msg = self.inbox.get()

            # One less is running.
            running -= 1
            
            # Get part.
            part = msg['part']
            print part, running

            # A message bring results and needed fridges.
            self.needed_Q[part] = self.needed_Q[part].union(set(msg['extra_Q'].keys()))

            # Update the Q values.
            self.Q[part].data.update(msg['own_Q'])

            # And we add possible jobs for the future.
            for result in msg['results']:
                state, action= result

                # If final state count iteration, if not save intermidiate job.
                if not state.is_terminal():
                    self.initial_states.append(result)
                else:

                    # Get the Q values from the initial state.
                    own_Q = self.Q[self.initial_state.part].data
                    extra_Q = dict((k, self.Q[k[0].part][k]) for k in self.needed_Q[self.initial_state.part])

                    # Send the job.
                    self.outbox[self.initial_state.part].put((self.initial_state, None, own_Q, extra_Q))
                    
                    running += 1
                    it += 1
                    print "Iteration: ", it                
                                                       

    def greedy_action(self, state, Epsilon=None):
        " Compute the E-greedy action. "

        # Get the current epsilon or the stored one (to control the ??greedyness??)
        Epsilon = Epsilon or self.Epsilon 

        # List of actions.
        actions = state.actions()

        # List of tuples (action, value)
        actions_value = [(a, self.Q[state.part][(state, a)]) for a in actions]            
        actions_value.sort(key=lambda x: -x[1])

        # We now have the state-actions values in order, add the probability.
        actions_p = [1 - Epsilon + Epsilon / len(actions)] + [Epsilon / len(actions)]

        # Check if we're greed in this case.
        s = numpy.random.random()
        if s < 1 - Epsilon:
            return actions_value[0][0]

        # Choose one between the rest.
        s -= 1 - Epsilon
        s *= (len(actions) - 1) / Epsilon
        return actions_value[1 + int(s)][0]

            
    def eval(self, max_iter=100):
        """ Just evaluate the current policy."""

        # Save the history in different array.
        history = [self.initial_state]
        state = self.initial_state
        reward = 0

        for it in range(max_iter):

            part = state.part

            # Complete greedy action, no exploration!.
            action = self.greedy_action(state=state, Epsilon=0)

            # Take the greedy action.
            state, r = state.next_state(action)
            reward += r

            # Save the state.
            history.append(state)

            if state.is_terminal():
                break

        return history, reward


class BarnState(object):
    """
    State in the Barn
    =================

    An state in the barn is defined by the position of the rat and the food still in the barn.

    :position: must be a 2-tuple of x,y coord.
    :food: a list of 2-tuples with the coord of the food.

    """

    def __init__(self, position, food, size, amount_food):
        self.position = position
        self.food = food
        self.size = size
        self.amount_food = amount_food
        
        self.part = len(food)
        self.parts = range(self.amount_food + 1)

    def __hash__(self):
        h = "%d:" % self.size
        h += "%d:%d:" % self.position
        h += ":".join("%d:%d" % f for f in self.food)
        return  hash(h)

    def __eq__(self, other):            
        return (self.position == other.position) and (self.food == other.food) and (self.size == other.size)

    def next_state(self, action):

        if action == 'right':
            next_position = ((self.position[0] + 1) % self.size, self.position[1])

        elif action == 'left':
            next_position = ((self.position[0] - 1) % self.size, self.position[1])

        elif action == 'up':
            next_position = (self.position[0], (self.position[1] + 1) % self.size)

        elif action == 'down':
            next_position = (self.position[0], (self.position[1] - 1) % self.size)

        elif action == 'stay':
            next_position = (self.position[0], self.position[1])

        next_food = [f for f in self.food if f != next_position]
        reward = -1 if (len(next_food) == len(self.food)) else 1

        if (next_food == []) and (next_position == (self.size-1, self.size-1)):
            reward = 10

        return BarnState(next_position,  next_food, self.size, self.amount_food), reward
            
    def is_terminal(self):
        return (self.food == []) and (self.position == (self.size-1, self.size-1))

    def actions(self):
        actions = ["up", "down", "left", "right", "stay"]

        if self.position[0] == 0:
            actions.pop(actions.index("left"))
        if self.position[1] == 0:
            actions.pop(actions.index("down"))
        if self.position[0] == self.size - 1:
            actions.pop(actions.index("right"))
        if self.position[1] == self.size - 1:
            actions.pop(actions.index("up"))

        return actions

    @classmethod
    def initial_state(cls, size):

        # Randomly locate the food on the ba<rn.
        amount_food = randint(size / 2, 2 * size)
        food = []

        while len(food) < amount_food:

            # Add a new piece of food.
            food.append((randint(0, size-1), randint(0, size-1)))

            # Ensure uniqueness.
            food = list(set(food) - set([(0, 0)]))

        return cls((0 ,0), food, size, amount_food)
        

def plot_evaluation(history, title, max_size, food, filename):
    " Plot an evaluation. "

    pylab.clf()
    pylab.xlim(-2, max_size + 1)
    pylab.ylim(-2, max_size + 1)
    pylab.axis('off')

    # Start.
    pylab.title(title)
    pylab.annotate('start', xy=(0, 0), xytext=(-1, -1), size=10, bbox=dict(boxstyle="round4,pad=.5", fc="0.8"), arrowprops=dict(arrowstyle="->"))
    pylab.annotate('stop', xy=(max_size-1, max_size-1), xytext=(max_size, max_size), size=10, bbox=dict(boxstyle="round4,pad=.5", fc="0.8"), arrowprops=dict(arrowstyle="->"))

    # Show the food.
    for f in food:
        pylab.annotate('food', xy=f, size=5, bbox=dict(boxstyle="round4,pad=.5", fc="0.8"), ha='center')

    # Create the x and y locations.
    x = [s.position[0] for s in history]
    y = [s.position[1] for s in history]    
    pylab.plot(x, y)

    # Save the figure
    pylab.savefig(filename)


if __name__ == '__main__':

    # Create the manager for the multiprprocessing.
    manager = Manager()

    # Size.
    size = 2

    # Intial state.
    initial_state = BarnState.initial_state(size)

    # Create a single sarsa.
    sarsa = Sarsa(manager, initial_state, Epsilon=0.8)

    # Iterate 1000 times using the pool.
    it = 0
    per_it = 10
    for i in range(100):
        sarsa.iterate(per_it)
        it += per_it
        history, reward = sarsa.eval()        
        plot_evaluation(history, "Parallel: iteration %d with reward %d" % (it, reward), size, initial_state.food, "parallel-iteration-%d.png" % (it, ))

