# Simulation constants

T = 100 # total number of time steps
W = 5 # prediction horizon
P = 3 # number of priorities (paper: 3)
M = 8 # number of OBP modules (paper: 16)

Q_BAR = 10*P # Maximum occupation of a given queue
Q_0 = 0 # Final occupation of queue
C_BAR = 10 # Maximum capacity, per module
DELTA_W_BAR = 0.1 # Maximum weight deviation
DELTA_S = 0.1
k_p = [10, 4, 1] # cost of packet loss. From highest to lowest priority (paper values)
W_VALUES = [2, 5, 10, 20]
NUM_SIM = 1


# Matrices constants

NUM_VAR_TYPES = 6
NUM_VAR = NUM_VAR_TYPES*M*P*(W+1)
NUM_EQ_CONSTRAINTS = 6
NUM_INEQ_CONSTRAINTS = 26
NUM_ROWS_PER_CONSTRAINT = M*P # Each constraint leads to M*P rows
NUM_COLS_PER_W = M * P * NUM_VAR_TYPES
SIZE_b = (M*P + M + P)*W + M*P*(W-1) + 2*M*P

f_in_low_bound          = 0*M*P
f_in_high_bound         = 1*M*P
w_low_bound             = 1*M*P
w_high_bound            = 2*M*P
L_low_bound             = 2*M*P
L_high_bound            = 3*M*P
f_out_low_bound         = 3*M*P
f_out_high_bound        = 4*M*P
Q_low_bound             = 4*M*P
Q_high_bound            = 5*M*P
delta_Q_low_bound       = 5*M*P
delta_Q_high_bound      = 6*M*P
Q_t_plus_1_low_bound    = Q_low_bound  + NUM_COLS_PER_W
Q_t_plus_1_high_bound   = Q_high_bound + NUM_COLS_PER_W
w_t_minus_1_low_bound   = 1*M*P - NUM_COLS_PER_W
w_t_minus_1_high_bound  = 2*M*P - NUM_COLS_PER_W