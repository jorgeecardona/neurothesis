"""
The Barn and the Rat
====================

First we create a squaed Barn with size n x n, and we place m pieces
of food on it.

From the barn we can get the number of possible states of the problem.
Given a barn of size n by n with m pices of food located on it, the number
of states of the system is given by all the possible locations of the rat
in the barn and the foo still in the barn, there are n^2 locations, and
there are 2^m states for the pieces of food, then roughlt speaking there are
n^2 * 2^m states. An state is defined by the location of the rat and the
pieces of food still in the barn, once we choose one location, there are
2 ^ m possibilities for the food. Also, if we choose a possition with food
(there are m of them) we can choose only 2 ^ (m-1), so the real number of states are:

 States: (n ^ 2 - m) * 2 ^ m + m * 2 ^ (m - 1) = (2 * n ^ 2 - m ) * 2 ^ (m -1)

Now, a policy is a map from the states to actions, we have at most 4 actions:
left, right, up and down. In some cases just 3 or 2.

We can compute the Value (in terms of the book) of each action given a state,
we estimate that the value is one in any state with no food,
and zero in any state with all the food on it.The Rat will learn to eat the food.

"""
import time
import numpy as np
from random import choice
import matplotlib.pyplot as plt
    

class Barn(object):

    def __init__(self, n, m, alpha=0.1, epsilon=0.1):

        # Save the epsilon.
        self.epsilon = epsilon

        # Save alpha parameter.
        self.alpha = alpha

        # Save the size and the food.
        self.size = n
        self.food_size = m

        # Locate the food.
        self.food = set()
        while len(self.food) < self.food_size:
            self.food.add(tuple(np.random.random_integers(0, self.size - 1, 2)))
        self.food = list(self.food)

        # Dynamically save the values.
        self._values = {}

        # Current state.
        self.current = "0:0:" + "1" * self.food_size

        # Remember the actions.
        self.positions = [(0, 0)]
        

    # def __repr__(self):
    #     # Create a nice representaiton of the barn.

    #     # Rat position.
    #     pos = int(self.current.split(':')[0])
    #     rat_i = pos / self.food_size
    #     rat_j = pos % self.food_size

    #     msg = "+" + "-" * 3 * self.size + "+\n"
        
    #     for i in range(self.size):
    #         msg += "|"
            
    #         for j in range(self.size):

    #             if rat_i == i and rat_j == j:
    #                 msg += " R "
    #             else:
    #                 msg += " * " if (self.matrix[i, j] == 1) else "   "

    #         msg += "|\n"

    #     msg += "+" + "-" * 3 * self.size + "+\n"
    #     msg += " Current State: %s\n" % self.current
    #     msg += " Number of states: %d\n" % ((2 * self.size ** 2 - self.food_size ) * 2 ** (self.food_size - 1))
    #     msg += " Actions: %s\n" % (", ".join(self.actions(self.current)))
    #     for action in self.actions(self.current):
    #         msg += "   %s: next:%s\n" % (action, self.next_state(action))
    #     return msg


    def value(self, id):
        " We encode any value as: {i * n + j}:{p0}{p1}{p2}... and so on, then "

        if id not in self._values:

            if id.endswith(":" + "0" * self.food_size):
                # There is no food in the barn.
                return 1.0

            if id.endswith(":" + "1" * self.food_size):
                # There is no food in the barn.
                return 0.0

            return 0.0

        return self._values[id]

    def update_value(self, id, correction):
        if id not in self._values:
            self._values[id] = 0.0
            
        self._values[id] += self.alpha * correction

    def actions(self, id):
        """ Given some state we compute the possible actions. """

        # Possible actions.
        actions = set(['left', 'right', 'up', 'down'])

        # Rat position.
        x, y = map(int, self.current.split(':')[:2])

        if x == 0:
            actions.remove('left')
        if y == 0:
            actions.remove('down')
        if x == self.size - 1:
            actions.remove('right')
        if y == self.size - 1:
            actions.remove('up')

        return actions              

    def next_state(self, action):
        """ Compute the next state given an action. """

        x, y, food = self.current.split(':')
        x = int(x)
        y = int(y)
        
        if action == 'left':
            x -= 1
        if action == 'right':
            x += 1
        if action == 'up':
            y += 1
        if action == 'down':
            y -= 1

        # Any food in the place.
        if (x, y) in self.food:
            index = self.food.index((x, y))
            food = food[:index] + '0' + food[index + 1:]

        return '%d:%d:%s' % (x, y, food)
                        
    def choose_action(self):
        """ Given the curent state we choose an action. """

        values = []
        
        for action in self.actions(self.current):

            # Compute the next state.
            next_state = self.next_state(action)

            # Get the value of this actions.
            values.append((action, next_state, self.value(next_state)))

        # Greedy guy.
        max_value = sorted(values, lambda x, y: int(y[2] - x[2]))[0][2]

        # Get the epsilon close states and choose one.
        states = [v for v in values if abs(v[2] - max_value) < self.epsilon]
        action, next_state = choice(states)[:2]

        # Update values.
        self.update_value(self.current, self.value(next_state) - self.value(self.current))        
        self.current = next_state

        # Store the action.
        self.positions.append(map(int, self.current.split(':')[:2]))

    def clear(self):
        " Clear the memory for a new run."

        # Current state.
        self.current = "0:0:" + "1" * self.food_size
        self.positions = []

    def run(self, maxit=1000):
        " Run until the rat finish the food"

        for i in range(maxit):
            self.choose_action()
            
            if self.current.endswith(":" + "0" * self.food_size):
                print "done!"
                break

        # Return the list of x, y of positions.
        x = []
        y = []
        for pos in self.positions:
            x.append(pos[0])
            y.append(pos[1])

        x = np.array(x)
        y = np.array(y)
        
        return x, y


if __name__ == "__main__":
    
    b = Barn(3, 2, epsilon=0.4)

    for i in range(100):
        
        x, y = b.run(1000)
        b.clear()

    x, y = b.run(1000)


    
    plt.figure()
    plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=10)
    x, y = zip(*b.food)
    plt.plot(x, y, 'ro')
    plt.show()
