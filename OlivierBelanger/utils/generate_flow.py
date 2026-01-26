import numpy as np
from oop_opt.constants import *

def generate_flows(flow_type='linear'):
    realized_flows, expected_flows = np.zeros((T+W, P)), np.zeros((T+W, P))

    if flow_type == 'linear':
        lambda_rate = 10
        increment = 0.2

        for t in range(T+W):
            if t < T:
                lambda_rate += increment
                lambda_rate_M = lambda_rate * M/2
                for p in range(P):
                    realized_flows[t, p] = np.random.poisson(lambda_rate_M/k_p[p])
                    expected_flows[t, p] = lambda_rate_M/k_p[p]
            else:
                for p in range(P):
                    realized_flows[t, p] = 0
                    expected_flows[t, p] = 0

    elif flow_type == 'mmpp':
        # MMPP
        # Transition matrix
        P_lambda = np.array([
            [0.8, 0.15, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.2, 0.75]
        ])

        lambdas = [10, 30, 50]

        current_state = 0  # Starting in state 0
        for t in range(T + W):
            if t < T:
                current_state = np.random.choice([0, 1, 2], p=P_lambda[current_state])
                lambda_rate_M = lambdas[current_state] * M / 2

                for p in range(P):
                    realized_flows[t, p] = np.random.poisson(lambda_rate_M / k_p[p])
                    expected_flows[t, p] = lambda_rate_M / k_p[p]
            else:
                for p in range(P):
                    realized_flows[t, p] = 0
                    expected_flows[t, p] = 0

    else:
        raise ValueError("Invalid flow_type. Choose either 'linear' or 'mmpp'.")

    return realized_flows, expected_flows