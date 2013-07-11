import time
from sarsa import State, Sarsa
from itertools import izip_longest
from random import randint
from math import log10
import pylab
import sys
import numpy


class BarnState(State):
    """
    State in the Barn
    =================

    An state in the barn is defined by the position of the rat and the food still in the barn.

    :position: must be a 2-tuple of x,y coord.
    :food: a list of 2-tuples with the coord of the food.

    """

    def __init__(self, position, food, max_size):
        self.position = position
        self.food = food

        self.max_size = max_size

    def __hash__(self):
        h = "%d:" % self.max_size
        h += "%d:%d:" % self.position
        h += ":".join("%d:%d" % f for f in self.food)
        return  hash(h)

    def __eq__(self, other):            
        return (self.position == other.position) and (self.food == other.food) and (self.max_size == other.max_size)

    def next_state(self, action):

        if action == 'right':
            next_position = ((self.position[0] + 1) % self.max_size, self.position[1])

        elif action == 'left':
            next_position = ((self.position[0] - 1) % self.max_size, self.position[1])

        elif action == 'up':
            next_position = (self.position[0], (self.position[1] + 1) % self.max_size)

        elif action == 'down':
            next_position = (self.position[0], (self.position[1] - 1) % self.max_size)

        next_food = [f for f in self.food if f != next_position]
        reward = -1 if (len(next_food) == len(self.food)) else 1

        return BarnState(next_position,  next_food, self.max_size), reward
            
    def is_terminal(self):
        return (self.food == []) and (self.position == (self.max_size-1, self.max_size-1))

    def actions(self):
        actions = ["up", "down", "left", "right"]

        if self.position[0] == 0:
            actions.pop(actions.index("left"))
        if self.position[1] == 0:
            actions.pop(actions.index("down"))
        if self.position[0] == self.max_size - 1:
            actions.pop(actions.index("right"))
        if self.position[1] == self.max_size - 1:
            actions.pop(actions.index("up"))

        return actions

    


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


if __name__ == "__main__":

    # We want 5 scenarios.
    number_of_scenarios = 5

    # Fixed size.
    max_size = 10

    # Global parameters.
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.2

    # Number of iterations.
    max_iters = 20000

    for n in range(1, number_of_scenarios + 1):

        # Randomly locate the food on the barn.
        amount_food = randint(max_size / 2, 2 * max_size)
        food = []

        while len(food) < amount_food:

            # Add a new piece of food.
            food.append((randint(0, max_size-1), randint(0, max_size-1)))

            # Ensure uniqueness.
            food = list(set(food))

        # Start the algorithm.
        sarsa = Sarsa(BarnState((0,0), food, max_size), epsilon=epsilon, alpha=alpha, gamma=gamma)
        sarsa.seed(int(100 * time.time()))

        # keep track of how much do we move the q.
        track = []

        for it in range(1, max_iters + 1):

            if it % 10 == 0:
                print "Scenario %d: %d/%d\r" % (n, it, max_iters) ,
                sys.stdout.flush()

            history, corrections = sarsa.iterate()
            track.append(numpy.sqrt(sum(map(lambda x: x*x, corrections))))
            
            # We're just selecting nice places to evaluate the current policy and create a picture.
            if (it % 10 ** int(log10(it)) == 0) and (it / 10 ** int(log10(it)) in [1, 2, 4, 8]):
                print " evaluationg current policy at %d ..." % it                    
                history, reward = sarsa.eval(max_size ** 2)
            
                # Plot this.
                plot_evaluation(history, "Scenario %d at iteration %d with reward %d" % (n, it, reward), max_size, food, "scenario-%d-iteration-%d.png" % (n, it))

        pylab.clf()
        pylab.plot(track)
        pylab.savefig("scenario-%d-learning.png" % (n, ))
                
            
