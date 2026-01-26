import numpy as np
import os
from oop_opt.constants import *
import pandas as pd

def kronecker_delta(i, j):
    """Kronecker delta function."""
    return int(i == j) 

def generate_v_m(m):
    """
    Generate a single row of the v_m pattern for a given module.
    """
    # Block for single module
    block = np.zeros(M)
    block[m-1] = 1  # Set m-th element to 1
    # Repeat block P times to create row
    row = np.tile(block, P)
    
    return row

def generate_v_p(p):
    row = np.zeros(M * P)
    # Set the elements from (p-1)*M to p*M to 1
    row[(p-1)*M : p*M] = 1

    return row

def generate_A_matrix(start_t, end_t, t, idx):

    if t == start_t:
        start_rows = [0, M*P, M*P + M, M*P + M + P, 2*M*P + M + P, 3*M*P + M + P]
        num_rows = 4*M*P + M + P
    else:
        if idx < W-1:
            start_rows = [0, M*P, M*P + M, M*P + M + P, 0, 0]
            num_rows = 2*M*P + M + P
        else:
            start_rows = [0, M*P, M*P + M, 0, 0, 0]
            num_rows = M*P + M + P
            
    num_cols = M * P * NUM_VAR_TYPES * (W+1)
    col_shift = idx * NUM_COLS_PER_W
    A = np.zeros((num_rows, num_cols))

    bounds = [
        (f_in_low_bound, f_in_high_bound, 1),
        (L_low_bound, L_high_bound, -1),
        (f_out_low_bound, f_out_high_bound, -1),
        (delta_Q_low_bound, delta_Q_high_bound, -1)
    ]

    
    #  Equality constraint 1: f_in - L - f_out - delta_Q = 0
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            for low_bound, high_bound, multiplier in bounds:
                if low_bound + col_shift <= col < high_bound + col_shift:
                    A[row, col + NUM_COLS_PER_W] = multiplier * kronecker_delta(row + low_bound, col_mod)
                    break
    
    # Equality constraint 2: sum(w) = 1 for a given m
    for m in range(1, M + 1):
        pattern_row = generate_v_m(m)
        row_idx = start_rows[1] + (m-1)
        col_start = M*P + col_shift + NUM_COLS_PER_W
        col_end = col_start + M*P
        A[row_idx, col_start:col_end] = pattern_row
    
    # Equality constraint 3: sum(f_in) = F for a given p
    for p in range(1, P + 1):
        pattern_row = generate_v_p(p)
        row_idx = start_rows[2] + (p-1)
        A[row_idx, col_shift + NUM_COLS_PER_W : M*P + col_shift + NUM_COLS_PER_W] = pattern_row

    # Equality Constraint 4: Q(t+1) = Q(t) + delta_Q
    if idx != W - 1:
        for row in range(NUM_ROWS_PER_CONSTRAINT):
            actual_row = start_rows[3] + row
            for col in range(num_cols):
                col += col_shift
                if Q_low_bound + col_shift <= col < Q_high_bound + col_shift:
                    col_mod = col % col_shift if col_shift != 0 else col
                    A[actual_row, col + NUM_COLS_PER_W] = kronecker_delta(row + Q_low_bound, col_mod)
                elif delta_Q_low_bound + col_shift <= col < delta_Q_high_bound + col_shift:
                    col_mod = col % col_shift if col_shift != 0 else col
                    A[actual_row, col + NUM_COLS_PER_W] = kronecker_delta(row + delta_Q_low_bound, col_mod)
                elif Q_t_plus_1_low_bound + col_shift <= col < Q_t_plus_1_high_bound + col_shift:
                    col_mod = (col % col_shift +NUM_COLS_PER_W) if col_shift != 0 else col
                    A[actual_row, col + NUM_COLS_PER_W] = -kronecker_delta(row + Q_t_plus_1_low_bound, col_mod)

    # Equality constraint 5: Q = Q_system
    if t == start_t:
        for row in range(NUM_ROWS_PER_CONSTRAINT):
            actual_row = start_rows[4] + row
            for col in range(num_cols):
                col += col_shift
                col_mod = col % col_shift if col_shift != 0 else col
                if Q_low_bound + col_shift <= col < Q_high_bound + col_shift:
                    A[actual_row, col + NUM_COLS_PER_W] = kronecker_delta(row + Q_low_bound, col_mod)

    # Equality constraint 6: w(t-1) = realization
    if t == start_t:
        for row in range(NUM_ROWS_PER_CONSTRAINT):
            actual_row = start_rows[5] + row
            for col in range(num_cols):
                col += col_shift
                col_mod = col % col_shift if col_shift != 0 else col
                if w_low_bound + col_shift <= col < w_high_bound + col_shift:
                    A[actual_row, col] = kronecker_delta(row + w_low_bound, col_mod)

    return A


def generate_b_vector(t, start_t, flows, Q_sim, w_t_minus_1_realised, optim_w_start_0):
    b1 = np.zeros((M * P, 1)) 
    b4 = np.zeros((M * P, 1)) if t < start_t + W - 1 else np.array([]).reshape(0, 1)
    b5 = np.array([]).reshape(0, 1)
    b6 = np.array([]).reshape(0, 1) 

    if t < T:
        b2 = np.ones((M, 1))
        b3 = flows.reshape(-1, 1)
    else:
        b2 = np.zeros((M, 1))
        b3 = np.zeros((P, 1))

    if t == start_t:
        b5 = np.full((M * P, 1), Q_sim) if isinstance(Q_sim, int) else Q_sim.reshape(P*M, 1)

        if t  == 0:
            b6 = np.full((M * P, 1), 0.5)
        else:
            b6 = np.full((M * P, 1), w_t_minus_1_realised) if isinstance(w_t_minus_1_realised, int) else w_t_minus_1_realised.reshape(P*M, 1)

    b = np.concatenate((b1, b2, b3, b4, b5, b6))
    return b


def generate_g_c_matrices(idx):
    
    # num_cols = M * P * NUM_VAR_TYPES * W_adjusted
    num_cols = M * P * NUM_VAR_TYPES * (W + 1)
    col_shift = idx * NUM_COLS_PER_W

    g_c_1 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_2 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_3 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_4 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_5 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_6 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_7 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))

    g_c_8 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_9 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_10 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_11 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_12 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_13 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_14 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_15 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_16 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_17 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_18 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))

    g_c_19 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_20 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_21 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_22 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_23 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_24 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_25 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_26 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_27 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_28 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))
    g_c_29 = np.zeros((NUM_ROWS_PER_CONSTRAINT, num_cols))

    
    # Inequality constraint 1: w >= 0
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if w_low_bound + col_shift <= col < w_high_bound + col_shift:
                g_c_1[row, col + NUM_COLS_PER_W] = -kronecker_delta(row + w_low_bound, col_mod)

    # Inequality constraint 2:                                                  w <= 1
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if w_low_bound + col_shift <= col < w_high_bound + col_shift:
                g_c_2[row, col + NUM_COLS_PER_W] = kronecker_delta(row + w_low_bound, col_mod)


    # Inequality constraint 3:                                                  w(t) - (w(t-1) + delta_w_bar)
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if w_t_minus_1_low_bound + col_shift <= col < w_t_minus_1_high_bound + col_shift:
                g_c_3[row, col] = -kronecker_delta(row + w_t_minus_1_low_bound, col_mod)
            if w_low_bound + col_shift <= col < w_high_bound + col_shift:
                g_c_3[row, col] = kronecker_delta(row + w_low_bound, col_mod)

    # Inequality constraint 4:                                                  -w(t) - (-w(t-1) + delta_w_bar)
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if w_t_minus_1_low_bound + col_shift <= col < w_t_minus_1_high_bound + col_shift:
                g_c_3[row, col] = kronecker_delta(row + w_t_minus_1_low_bound, col_mod)
            if w_low_bound + col_shift <= col < w_high_bound + col_shift:
                g_c_3[row, col] = -kronecker_delta(row + w_low_bound, col_mod)


    # Inequality constraint 5:                                                  f_out - w/delta_S
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if w_low_bound + col_shift <= col < w_high_bound + col_shift:
                g_c_5[row, col + NUM_COLS_PER_W] = -1/DELTA_S * kronecker_delta(row + w_low_bound, col_mod)
            if f_out_low_bound + col_shift <= col < f_out_high_bound + col_shift:
                g_c_5[row, col + NUM_COLS_PER_W] = kronecker_delta(row + f_out_low_bound, col_mod)


    # Inequality constraint 6:                                                  sum(Q) - Q_bar
    for m in range(1, M + 1):
        pattern_row = generate_v_m(m)
        for p in range(P):
            col_start = Q_low_bound + col_shift
            col_end = col_start + M*P
            row_idx = (m-1)*P + p
            g_c_6[row_idx, col_start:col_end] = pattern_row
    
    # Inequality constraint 7:                                                  sum(Q + delta_Q) - Q_bar
    for m in range(1, M + 1):
        pattern_row = generate_v_m(m)
        for p in range(P):
            for low_bound in [Q_low_bound, delta_Q_low_bound]:
                col_start = low_bound + col_shift + NUM_COLS_PER_W
                col_end = col_start + M*P
                row_idx = (m-1)*P + p
                g_c_7[row_idx, col_start:col_end] = pattern_row



    # Inequality constraint 8:                                                  f_in >= 0
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if f_in_low_bound + col_shift <= col < f_in_high_bound + col_shift:
                g_c_8[row, col] = -kronecker_delta(row + f_in_low_bound, col_mod)
    
    # Inequality constraint 9:                                                  f_out >= 0
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if f_out_low_bound + col_shift <= col < f_out_high_bound + col_shift:
                g_c_9[row, col] = -kronecker_delta(row + f_out_low_bound, col_mod)
    
    # Inequality constraint 10:                                                  L >= 0
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if L_low_bound + col_shift <= col < L_high_bound + col_shift:
                g_c_10[row, col] = -kronecker_delta(row + L_low_bound, col_mod)
    
    # Inequality constraint 11:                                                  Q >= 0
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if Q_low_bound + col_shift <= col < Q_high_bound + col_shift:
                g_c_11[row, col] = -kronecker_delta(row + Q_low_bound, col_mod)

    # Inequality constraint 12:                                                  delta_Q >= -100
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if delta_Q_low_bound + col_shift <= col < delta_Q_high_bound + col_shift:
                g_c_12[row, col + NUM_COLS_PER_W] = -kronecker_delta(row + delta_Q_low_bound, col_mod)


    
    # Inequality constraint 13:                                                  f_in <= 100
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if f_in_low_bound + col_shift <= col < f_in_high_bound + col_shift:
                g_c_13[row, col] = kronecker_delta(row + f_in_low_bound, col_mod)
    
    # Inequality constraint 14:                                                  f_out <= 100
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if f_out_low_bound + col_shift <= col < f_out_high_bound + col_shift:
                g_c_14[row, col] = kronecker_delta(row + f_out_low_bound, col_mod)
    
    # Inequality constraint 15:                                                  L <= 100
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if L_low_bound + col_shift <= col < L_high_bound + col_shift:
                g_c_15[row, col] = kronecker_delta(row + L_low_bound, col_mod)
    
    # Inequality constraint 16:                                                  Q <= 100
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if Q_low_bound + col_shift <= col < Q_high_bound + col_shift:
                g_c_16[row, col] = kronecker_delta(row + Q_low_bound, col_mod)
    
    # Inequality constraint 17:                                                  delta_Q <= 100
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if delta_Q_low_bound + col_shift <= col < delta_Q_high_bound + col_shift:
                g_c_17[row, col] = kronecker_delta(row + delta_Q_low_bound, col_mod)


    # Inequality constraint 18:                                                  w(t-1) >= 0
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if w_low_bound + col_shift <= col < w_high_bound + col_shift:
                g_c_18[row, col - NUM_COLS_PER_W] = -kronecker_delta(row + w_low_bound, col_mod)

    # Inequality constraint 19:                                                  f_in(t+1) >= 0
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if f_in_low_bound + col_shift <= col < f_in_high_bound + col_shift:
                g_c_19[row, col + NUM_COLS_PER_W] = -kronecker_delta(row + f_in_low_bound, col_mod)
    
    # Inequality constraint 20:                                                  f_out(t+1) >= 0
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if f_out_low_bound + col_shift <= col < f_out_high_bound + col_shift:
                g_c_20[row, col + NUM_COLS_PER_W] = -kronecker_delta(row + f_out_low_bound, col_mod)
    
    # Inequality constraint 21:                                                  L(t+1) >= 0
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if L_low_bound + col_shift <= col < L_high_bound + col_shift:
                g_c_21[row, col + NUM_COLS_PER_W] = -kronecker_delta(row + L_low_bound, col_mod)
    
    # Inequality constraint 22:                                                  Q(t+1) >= 0
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if Q_low_bound + col_shift <= col < Q_high_bound + col_shift:
                g_c_22[row, col + NUM_COLS_PER_W] = -kronecker_delta(row + Q_low_bound, col_mod)

    # Inequality constraint 23:                                                  delta_Q(t+1) >= -100
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if delta_Q_low_bound + col_shift <= col < delta_Q_high_bound + col_shift:
                g_c_23[row, col + NUM_COLS_PER_W] = -kronecker_delta(row + delta_Q_low_bound, col_mod)


    
    # Inequality constraint 24:                                                  f_in(t+1) <= 100
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if f_in_low_bound + col_shift <= col < f_in_high_bound + col_shift:
                g_c_24[row, col + NUM_COLS_PER_W] = kronecker_delta(row + f_in_low_bound, col_mod)
    
    # Inequality constraint 25:                                                  f_out(t+1) <= 100
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if f_out_low_bound + col_shift <= col < f_out_high_bound + col_shift:
                g_c_25[row, col + NUM_COLS_PER_W] = kronecker_delta(row + f_out_low_bound, col_mod)
    
    # Inequality constraint 26:                                                  L(t+1) <= 100
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if L_low_bound + col_shift <= col < L_high_bound + col_shift:
                g_c_26[row, col + NUM_COLS_PER_W] = kronecker_delta(row + L_low_bound, col_mod)
    
    # Inequality constraint 27:                                                  Q(t+1) <= 100
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if Q_low_bound + col_shift <= col < Q_high_bound + col_shift:
                g_c_27[row, col + NUM_COLS_PER_W] = kronecker_delta(row + Q_low_bound, col_mod)
    
    # Inequality constraint 28:                                                  delta_Q(t+1) <= 100
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if delta_Q_low_bound + col_shift <= col < delta_Q_high_bound + col_shift:
                g_c_28[row, col + NUM_COLS_PER_W] = kronecker_delta(row + delta_Q_low_bound, col_mod)


    # Inequality constraint 29:                                                  w(t+1) >= 0
    for row in range(NUM_ROWS_PER_CONSTRAINT):
        for col in range(num_cols):
            col += col_shift
            col_mod = col % col_shift if col_shift != 0 else col
            if w_low_bound + col_shift <= col < w_high_bound + col_shift:
                g_c_29[row, col + NUM_COLS_PER_W] = -kronecker_delta(row + w_low_bound, col_mod)



    return g_c_1, g_c_2, g_c_7, g_c_5, g_c_3, g_c_4, g_c_8, g_c_9, g_c_10, g_c_11, g_c_13, g_c_14, g_c_15, g_c_16, g_c_17, g_c_18, g_c_19, g_c_20, g_c_21, g_c_22, g_c_24, g_c_25, g_c_26, g_c_27, g_c_28, g_c_29


def generate_g_d_matrices():
    # values = [0, 1, 0, Q_BAR, Q_BAR]
    values = [0, 1, Q_BAR, 0, DELTA_W_BAR, DELTA_W_BAR, 0, 0, 0, 0, 50, 50, 50, 50, 100, 0, 0, 0, 0, 0, 50, 50, 50, 50, 100, 0]
    matrices = []

    for _, value in enumerate(values):
        g_d = np.zeros((NUM_ROWS_PER_CONSTRAINT, 1))
        g_d[:, 0] = value
        matrices.append(g_d)


    return matrices


def generate_c_vector(t):
    # W_adjusted = min(W, T - t)
    # num_cols = M * P * NUM_VAR_TYPES * W_adjusted 

    # c = np.zeros((1, num_cols))
    
    # for w in range(W_adjusted):
    #     for col in range(int(num_cols/W_adjusted)):
    #         shifted_col = col + int(w*num_cols/W_adjusted)
    #         if L_low_bound <= col < L_high_bound:
    #             k_p_index = (col - L_low_bound) // M 
    #             c[0, shifted_col] = k_p[k_p_index]

    num_cols = M * P * NUM_VAR_TYPES * (W+1) 

    c = np.zeros((1, num_cols))
    
    for w in range(W):
        for col in range(int(num_cols/W)):
            shifted_col = col + int(w*num_cols/(W+1))
            if L_low_bound <= col < L_high_bound:
                k_p_index = (col - L_low_bound) // M 
                c[0, shifted_col + NUM_COLS_PER_W] = k_p[k_p_index]
    return c

def save_to_file(filename, data, print_indexes = None, header=""):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'w') as f:
        for index, item in enumerate(data):
            if header:
                f.write(f'{header} {print_indexes[index]}:\n')
            np.savetxt(f, item, fmt='%.1f')
            f.write('\n')