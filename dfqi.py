import random
import model
from settings import DEVICE


class DFQI():

    def __init__(self):
        self.agent = model.Agent(DEVICE)

        # F|A| = Deep neural net representing state -> action space
        self.f_a = None

        # K = num of iterations of DFQI
        self.K = 5

        # N_epoch = epochs per iteration
        self.N_epoch = 20

        # b = mini-batch size
        self.b = 64

        # N_samples = num of samples
        self.N_samples = 1000

        self.gamma = 1

        self.Y = []
        self.D_n = []

    def init_neural_net(self):
        # Initialize neural network of Q_0 is elem of f_a
        self.Q_0 = ... #Neural net

    def generate_samples(self):
        # Generate samples D_n = {(X_i, A_i, R_i, X'_i)}
        # X_i is the state at time i
        # A_i is the sample action taken at X_i
        # R_i is the associated reward ~ R(X_i, A_i)
        # X'_i is the next state

        self.D_n = ...

    def yi(self):
        # Y_i = R_i + gamma*max(Q_k(X'_i, a') for i in 1,n))
        for i in range(0, self.N_samples):
            self.Y[i] = self.D_n[i][2] + self.gamma

    def define_regression_set(self):
        self.Y = 3
        self.Dp_n = [((X_i, A_i), Y_i) for Y_i, (X_i, A_i, _, _) in zip(self.Y, self.D_n)]

    def get_random_batch(self):
        random.shuffle(self.Dp_n)
        self.D_b = self.Dp_n[:self.b]

    def run_dfqi(self):
        for k in range(0, self.K-1):
            self.generate_samples()
            self.yi()
            self.define_regression_set()
            for _ in range(self.N_epoch):
                self.get_random_batch()
