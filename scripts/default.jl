
using CerebellarMotorLearning
using Distributions
using Plots
using Random
using LaTeXStrings
using LinearAlgebra
using LsqFit
using Printf
using DrWatson
using Flux

# Plots.default(titlefont = (20, "times"), legendfontsize = 18, guidefont = (18, :black), tickfont = (12, :black), grid=false, framestyle = :zerolines, yminorgrid = false, markersize=6)
Plots.default( grid=false, markersize=6,linewidth=3)
pwidth=1000
pheight=600

verbose = false # save for every size and every update 
preTrain = true

trainErrorIndex = 1
onlineErrorIndex = 1
outputIndex = 3
trajErrorIndex=2; # index of error in traj in ode system for training 

fc = 0.1; # cutoff frequency
Nref = 200 # for truncated fourier series

# dtI = (2*pi)/omega_c*0.2
num_nn_inputs = 10
num_nn_outputs = 1;

dtI = 0.2/fc # 0.2 of the fastest possible period
ddtI = dtI/(num_nn_inputs-1)
lookahead_times = 0.:ddtI:dtI

N = 1*num_nn_inputs; 
nnDims = (num_nn_inputs,N,num_nn_outputs)
K = 4;

trajTime = 1000.; # time of the reference trajectory for training 
trajTimeSS = 1000.; # traj time for ss calculation
trajTimePT = 7000. # pretrain trajectory time

trajTimePP = 500.; # traj time for task loss calculation

Ks = (0.1, 0.1, 0.1, 0.1);
# Ks = (0.0, 0.0, 0.0, 0.1);

# parameters for plant
A = [0 1 0 0; -2 -2 1 0; 0 0 0 1; 0 0 -2 -2];
B = reshape([0 ; 1 ; 0 ; 1 ],4,1);
C = reshape([1 0 0 0],1,4)
D = reshape([0],1,1)
plantMatrices = (A,B,C,D)

# OrnsteinUhlenbeck process parameters
θ = 0.1 # mean reverting parameter
μ = 0.0 # asymptotic mean 
sigma = 0.02 # random schocks 
# T = 100.0
dt = 0.01

# Training parameters
dt = 1.0
t_train = 0.01:dt:trajTime # train times (callbacks doesn   't stop at t=0.0)

deltaTe = 1.0
deltaTr = 0.5
deltaTh = 0.1

t_trainSS = 0.01:dt:trajTimeSS


