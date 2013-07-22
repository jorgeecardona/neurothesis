import sys
import numpy
import numpy.random
import random
from UserDict import UserDict
from multiprocessing import Process
from multiprocessing import Manager
import time
from state import BarnState
import pylab



class StateActionValues(UserDict):
    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = 0
        return self.data[key]    

    def update(self, data):
        return self.data.update(data)

class SarsaBase(object):

    def greedy_action(self, state, Epsilon=None):
        " Compute the E-greedy action. "

        # Get the current epsilon or the stored one (to control the ??greedyness??)
        Epsilon = Epsilon or self.Epsilon 

        # List of tuples (action, value)
        actions = [(a, self.Q[(state, a)]) for a in state.actions]
        actions.sort(key=lambda x: -x[1])

        # Check if we're greed in this case.
        s = numpy.random.random()
        if s <= 1 - Epsilon:
            return actions[0][0]

        # Choose one between the rest.
        s -= 1 - Epsilon
        s *= (len(actions) - 1) / Epsilon
        return actions[1 + int(s)][0]



class SarsaAgent(Process, SarsaBase):
    " All the sarsa agents are processes."

    def __init__(self, part, inbox, outbox, max_it=500, Epsilon=0.8, Alpha=0.1, Gamma=0.8):
        Process.__init__(self)

        # Save the parameters.
        self.Epsilon = Epsilon
        self.Alpha = Alpha
        self.Gamma = Gamma

        # Keep running flag.
        self.keep_running = True

        # Inbox to receive the jobs.
        self.inbox = inbox

        # Outbox.
        self.outbox = outbox

        # Save the own part.
        self.part = part

        # State value.
        self.Q = StateActionValues()

        # Maximum amount of iteration per time.
        self.max_it = max_it


        self.running_time = 0

    def iterate(self, state, action):

        # Take the greedy action.
        next_state, reward = state.next_state(action)

        # Second action.
        next_action = self.greedy_action(next_state)

        # Next Q(state, action).
        next_q = self.Q[(next_state, next_action)]

        # If the next state is in another part we use an estimate for the reward.
        if next_state.part != self.part:
            reward = next_q

        # Update Q(first_state, first_action)
        correction = reward + self.Gamma * next_q - self.Q[(state, action)]
        self.Q[(state, action)] += self.Alpha * correction

        return next_state, next_action

    def handle_control(self, control):

        if control == 'stop':
            self.keep_running = False

        if 'seed' in control:
            numpy.random.seed(control['seed'])
            
    def handle_problem(self, problem):
        " Handle the problem. "

        # print("Problem received for part: %d %s" % (self.part, problem))

        # Get the fringe data.
        self.Q.update(problem['Q'])

        it = 0

        start = time.clock()

        jobs = []

        while it < self.max_it:

            # Get the state-action.
            state = problem['state']
            action = problem['action']
            
            while state.part == self.part and not state.is_terminal:

                # Iterate
                state, action = self.iterate(state, action)

                it += 1

            if not state.is_terminal:
                # Send a new job.
                jobs.append((state.part, state, action))
            else:            
                # Send a done message.
                self.outbox.put({'done': state})

        # Send the jobs.        
        self.send_jobs(jobs)

        # Save the running time.
        self.running_time += time.clock() - start

    def send_jobs(self, jobs):
     
        fringe = {}

        for s, a in self.Q.data:

            if s.part != self.part:
                continue

            for sn in s.neighbors():
                if sn.part != self.part:
                    if sn.part not in fringe:
                        fringe[sn.part] = {}
                    fringe[sn.part][(s, a)] = self.Q[(s, a)]

        # Own Q.
        own_Q = dict(((s, a), self.Q[(s, a)]) for s, a in self.Q.data if s.part == self.part)

        msg = {
            'problem': {
                'jobs': jobs,
                'fringe': fringe,
                'Q': own_Q
                }
            }

        # Send a new job.
        print "Running Time %d: %f" % (self.part, self.running_time)
        self.outbox.put(msg)
            
    def run(self):

        print("Start agent for part: %d" % self.part)

        # Start requesting a problem.
        self.outbox.put({'request': {'part': self.part}})

        while self.keep_running:

            # Get a job to process.
            job = self.inbox.get()
                    
            if 'control' in job:
                self.handle_control(job['control'])
                continue

            if 'problem' in job:
                self.handle_problem(job['problem'])
                self.outbox.put({'request': {'part': self.part}})


class Sarsa(SarsaBase):
    
    def __init__(self, initial_state, Epsilon=0.7, Alpha=0.2, Gamma=0.5, prefix=None):

        # Save the parameters.
        self.Epsilon = Epsilon
        self.Alpha = Alpha
        self.Gamma = Gamma

        # Save the initial state.
        self.initial_state = initial_state

        # Save the parts.
        self.parts = self.initial_state.parts

        manager = Manager()

        # Create the inbox.
        self.inbox = manager.Queue()

        # Create the boxes.
        self.outboxes = dict((p, manager.Queue()) for p in self.parts)

        # Create the agents.
        self.agents = dict((p, SarsaAgent(p, self.outboxes[p], self.inbox, Epsilon=Epsilon, Gamma=Gamma, Alpha=Alpha)) for p in self.parts)

        # Keep running flag.
        self.keep_running = True

        # Keep track of who is requesting problems.
        self.requests = []        

        # Current jobs.
        self.jobs = dict((p, []) for p in self.parts)

        # State-Action values in the boundaries.
        self.fringe_Q = dict((p, {}) for p in self.parts)

        # Own state-action for evaluation.
        self.Q = StateActionValues()

        # Save prefix for created files.
        self.prefix = prefix or str(int(time.time()))

    def seed_all(self):
        for outbox in self.outboxes.values():
            outbox.put({'control': {'seed': int(time.time() * 1000000)}})

    def handle_problem(self, problem):

        # Save the fringe data.
        for p in problem['fringe']:
            self.fringe_Q[p].update(problem['fringe'][p])

        # Update the local copy of Q.
        self.Q.update(problem['Q'])

        # Save jobs.
        for job in problem['jobs']:
            part, state, action = job
            self.jobs[part].append({'state': state, 'action': action})

            # In case there is someone waiting for this job send it immediately.
            if part in self.requests:
                self.send_job(part)
                        
    def send_job(self, part):

        if part in self.requests:
            self.requests.pop(self.requests.index(part))
        
        # Select a job randomly. Choose between the last 10
        job = random.choice(self.jobs[part][-10:])

        # print("Send job to %d" % part)
        self.outboxes[part].put({
            'problem': {
                'action': job['action'],
                'state': job['state'],
                'Q': self.fringe_Q[part]
                }
            })    

    def handle_request(self, request):

        part = request['part']

        if len(self.jobs[part]) > 0:
            self.send_job(part)
        else:
            self.requests.append(part)

    def add_initial_job(self):
        self.jobs[self.initial_state.part].append({'state': self.initial_state, 'action': random.choice(self.initial_state.actions)})

    def run(self):

        start = time.clock()

        # Start all the agents.
        for agent in self.agents.values():
            agent.start()

        # Add an initial_state problem.
        self.add_initial_job()

        # Count the done.
        done_counter = 0

        while self.keep_running:

            # Get a message.
            msg = self.inbox.get()

            if 'request' in msg:                
                self.handle_request(msg['request'])

            if 'problem' in msg:
                self.handle_problem(msg['problem'])

            if 'done' in msg:
                
                done_counter += 1

                self.add_initial_job()

                print done_counter
                if done_counter % 10 == 0:
                    self.eval(done_counter)

    def greedy_action(self, state):
        " Compute the E-greedy action. "

        # List of tuples (action, value)
        actions = [(a, self.Q[(state, a)]) for a in state.actions]
        actions.sort(key=lambda x: -x[1])
        return actions[0][0]

    def eval(self, counter, max_it=100):
        " Do an evaluation with the current values."

        #     print "(%d, %d) %s %2.1f" % (k[0].position[0], k[0].position[1], k[1], v)
        # print ""

        state = self.initial_state
        it = 0
        
        history = [state]
        reward_total = 0

        while not state.is_terminal and it < self.initial_state.size ** 2:
            action = self.greedy_action(state)
            state, reward = state.next_state(action)
            reward_total += reward
            history.append(state)
            it += 1

        print reward_total, [s.position for s in history]
            
        pylab.clf()
        pylab.xlim(-2, self.initial_state.size + 1)
        pylab.ylim(-2, self.initial_state.size + 1)
        pylab.axis('off')

        # Start.
        pylab.title("Iteration: %d (Size %d with reward %d)" % (counter, self.initial_state.size, reward_total))
        pylab.annotate('start', xy=(0, 0), xytext=(-1, -1), size=10, bbox=dict(boxstyle="round4,pad=.5", fc="0.8"), arrowprops=dict(arrowstyle="->"))
        pylab.annotate('stop', xy=(self.initial_state.size-1, self.initial_state.size-1), xytext=(self.initial_state.size, self.initial_state.size), size=10, bbox=dict(boxstyle="round4,pad=.5", fc="0.8"), arrowprops=dict(arrowstyle="->"))

        # Show the food.
        for f in self.initial_state.food:
            pylab.annotate('food', xy=f, size=5, bbox=dict(boxstyle="round4,pad=.5", fc="0.8"), ha='center')

        # Create the x and y locations.
        x = [s.position[0] for s in history]
        y = [s.position[1] for s in history]    
        pylab.plot(x, y)

        # Save the figure        
        pylab.savefig("%s-%d.png" % (self.prefix, counter))
    
        

if __name__ == '__main__':

    # Size.
    size = int(sys.argv[1])
    prefix = sys.argv[2]

    # Intial state.
    initial_state = BarnState.initial_state(size)

    # Create a single sarsa.
    sarsa = Sarsa(initial_state, Epsilon=0.6, Alpha=0.2, Gamma=0.8, prefix=prefix)

    # Seed all the agents.
    sarsa.seed_all()

    sarsa.run()
