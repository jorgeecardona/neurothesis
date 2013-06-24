import time
from sarsa import State, Sarsa
from itertools import izip_longest
import pylab

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

def plot_path(path):

    pylab.clf()
    pylab.xlim(-2, max_size + 1)
    pylab.ylim(-2, max_size + 1)
    pylab.axis('off')

    # Start.
    pylab.annotate('start', xy=(0, 0), xytext=(-1, -1), size=10, bbox=dict(boxstyle="round4,pad=.5", fc="0.8"), arrowprops=dict(arrowstyle="->"))
    pylab.annotate('stop', xy=(9, 9), xytext=(10, 10), size=10, bbox=dict(boxstyle="round4,pad=.5", fc="0.8"), arrowprops=dict(arrowstyle="->"))

    # Show the food.
    for f in food:
        pylab.annotate('food', xy=f, size=5, bbox=dict(boxstyle="round4,pad=.5", fc="0.8"), ha='center')
    
    
    for i in range(len(path) - 1):
        pylab.arrow(path[i][0], path[i][1], path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])


# Parameters.
max_size = 10
food = [(0,8), (4,4)]

# Start the algorithm.
sarsa = Sarsa(BarnState((0,0), food, max_size), epsilon=0.05, alpha=0.2, gamma=0.1)
sarsa.seed(int(100* time.time()))

plot_in = [10, 100, 200, 400, 600, 1000, 1500, 2000, 4000, 5000, 6000, 8000, 10000] 
for i in range(max(plot_in) + 1):
    sarsa.iterate()

    if i % 10 == 0:
        print i
    
    if i in plot_in:
        plot_path([s.position for s in sarsa.history])
        pylab.savefig('/tmp/simple-path-%d.png' % i)
        print i


