import numpy as np
import pandas as pd


def get_state_reference(T, K):
    """Generates a list of K EVs and as possible reference signal the population may follow

    Args:
        T (int): Length of time horizon
        K (int): Number of EVs in popoulation

    """
    tasks = generate_population(T, K)
    state = get_state(tasks)
    reference = get_reference(tasks, T)
    return state, reference


def generate_population(T: int, K=20):
    """Generate feasible tasks

    Args:
        T (int): Time horizon
        K (int, optional): Population number. Defaults to 20.
        m_low (float, optional): Low bound on max capacity. Defaults to 0.5.
        m_high (float, optional): High bound on max capacity. Defaults to 1.5.

    Returns:
        _type_: E,a,d,m
    """

    a = np.zeros(K)  # arrival times
    d = np.random.randint(a, T-1)  # departure time (greater than or equal to arrival)
    m = np.ones(K)  # uniformly distributed max charging rate (normalise for number of vehicles in population)
    E = np.random.uniform(0, (d - a + 1) * m)  # uniform energy requirment whilst being feasible

    tasks = {
        'E': E,
        'a': a,
        'd': d,
        'm': m,
    }
    return tasks


def get_nu_dict(tasks):
    task_df = pd.DataFrame(tasks)
    task_df['q'] = np.floor(task_df['E'] / task_df['m'])

    T = np.max(tasks['d']) + 1
    null_df = get_null_df(T)
    task_df = pd.concat([task_df, null_df])

    task_df['r'] = task_df['E'] - task_df['q'] * task_df['m']
    task_df['p'] = task_df['d'] - task_df['a'] - task_df['q']
    cumulative_quotient = task_df.groupby(['a', 'd', 'p']).sum().groupby(level=[0, 1]).cumsum()['m']
    remainder = task_df.groupby(['a', 'd', 'p']).sum()['r']

    quotient_dict = cumulative_quotient.groupby(['a', 'd']).apply(list).to_dict()
    remainder_dict = remainder.groupby(['a', 'd']).apply(list).to_dict()

    nu_dict = {}
    for key in quotient_dict.keys():
        nu_dict[key] = remainder_dict[key] + np.pad(quotient_dict[key], (1, 0), constant_values=0)[:-1]

    return nu_dict


def get_double_stochastic(n, n_perms=10):
    np.random.seed(0)
    I = np.eye(n)
    B = np.zeros((n, n))
    lmda = np.random.uniform(0, 1, n_perms)
    lmda /= np.sum(lmda)
    for l in lmda:
        B += l * I[np.random.permutation(n)]
    return B


def get_reference(tasks, T, seed=0):
    np.random.seed(seed)
    nu_dict = get_nu_dict(tasks)
    x = np.zeros(T)
    a = 0
    for d in range(a, T):
        n = d - a + 1
        nu = nu_dict.get((a, d), np.zeros(n))

        B = get_double_stochastic(n)

        eye = np.eye(T, n, -a)
        x_ad = B @ nu
        x += eye @ x_ad
    return x


def get_state(tasks):
    df = pd.DataFrame(tasks)
    df['t_d'] = df['d'] + 1
    df['t_s'] = df['E'] / df['m']
    return df[['t_d', 't_s']]


def get_null_df(T):
    E = []
    a = []
    d = []
    m = []
    q = []

    for A in range(T):
        for D in range(A, T):
            for Q in range(D - A + 1):
                a.append(A)
                d.append(D)
                E.append(0)
                m.append(0)
                q.append(Q)
    null_tasks = {
        'E': E,
        'a': a,
        'd': d,
        'm': m,
        'q': q,
    }

    return pd.DataFrame(null_tasks)
