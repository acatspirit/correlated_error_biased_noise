import CompassCodes as cc
from compass_code_correlated_error import decoding_failures_correlated, decoding_failures_total


d = 9
l = 3

p =0.01
eta = 0.5
num_shots = 10_000

compass_code = cc.CompassCode(d=d, l=l)
H_x, H_z = compass_code.H['X'], compass_code.H['Z']
log_x, log_z = compass_code.logicals['X'], compass_code.logicals['Z']


%timeit decoding_failures_correlated(H_x, H_z, log_x, log_z, p, eta, num_shots)
%timeit decoding_failures_total(H_x, H_z, log_x, log_z, p, eta, num_shots)