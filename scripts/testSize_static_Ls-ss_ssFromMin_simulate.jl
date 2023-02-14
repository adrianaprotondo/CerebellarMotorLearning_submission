#####
# Test learning speed and steady state loss from task loss 
# for different NN sizes Ns and different learning steps,
# the bounds of the learning steps can vary with net size to make sure the system is learning (converging) 
# Train each system with LMS-like learning rule
# obtain ss from ss when initial weights are close to min 
# pretrain each size net with grad descent to get to some weight near local min
# save with drWatson the values
#####

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

using DataFrames, StatsPlots

include("sizeSim_functions.jl")

# simModel = "lms_size_Ls-ss"
simModel = "lms_size_Ls-ss_fourrier_o-u_2"
verbose = false # save for every size and every update in case it crashes (if true a lot more storage is needed)
preTrain = true # if true pre-train NN with gradient descent to get close to a local minimum for computing the ss loss

Plots.default(grid=false,linewidth=3)

####
# Define system parameters
####
trajErrorIndex=2; # index of error in traj in ode system for training 

fc = 0.1; # cutoff frequency
Nref = 200 # for truncated fourier series

# dtI = (2*pi)/omega_c*0.2
num_nn_inputs = 10
num_nn_outputs = 1;

dtI = 0.2/fc # 0.2 of the fastest possible period
ddtI = dtI/(num_nn_inputs-1)
# lookahead_times = 0.:0.1*0.5/20:0.9*0.5/20
lookahead_times = 0.:ddtI:dtI
# num_nn_inputs = length(lookahead_times)

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

####
# Define parameters for number of simulations and network sizes
####
numSim = 2; # number of different simulations (different initialisations)
seed = 32; # random seed to start with 
simI = seed:1:numSim+seed-1 # seeds to test
# Ns = 1*num_nn_inputs:1*num_nn_inputs:9*num_nn_inputs; # number of hidden units in the network to test
Ns = 1*num_nn_inputs:1*num_nn_inputs:9*num_nn_inputs; # number of hidden units in the network to test
trajTimePTs = [trajTimePT*(Ns[end]+1-N)^0.1 for N in Ns ] # define pre-train times for each sized net. Smaller nets are trained for longer to make sure they get close to local minimum 


################  
# Define systems
################
# OrnsteinUhlenbeck process parameters
θ = 0.1 # mean reverting parameter
μ = 0.0 # asymptotic mean 
sigma = 0.02 # random schocks 
# T = 100.0
dt = 0.01

# generate reference trajectory as long as the longest trajectory 
maxTime = maximum(vcat(trajTimePTs,trajTime,trajTimeSS,trajTimePP))

systemsSim = map(simI) do r
    # refF = refFourier(fc,maxTime)
    refF =  refFourier(fc,Nref,"o-u",[1/Nref,2*pi*Nref])
    # refF = ouFunDef(θ,sigma,μ,maxTime+dtI,dt,fc*dt) # use filtered O-U process as reference function 
    # plot(1:trajTime,refF(1:trajTime),lw=3)
    # refF = sinFunDef(fc) # Use sum of sins as reference function
    build_systems_sim(Ns,plantMatrices,Ks,refF,trajTime,lookahead_times,num_nn_inputs,num_nn_outputs,K,r);  
end
trainErrorIndex = systemsSim[1][1].trainErrorIndex 


################  
# Train parameters
################
dt = 1
t_train = 0.01:dt:trajTime # train times (callbacks doesn't stop at t=0.0)
deltaTe = 1.0
deltaTr = 0.5
deltaTh = 0.1
t_trainSS = 0.01:dt:trajTimeSS

# mus = collect(0.01:0.02:0.8) # learning steps
# mus = collect(0.01:0.02:0.2) # learning steps
mus = collect(0.01:0.01:0.1) # learning steps
# mus = [0.001,0.005,0.01,0.02,0.05,0.08,0.1,0.15,0.2,0.5]
muScale = 0.5 # scaling parameter of learning steps for different sized nets
qs = Ns./Ns[1] # expansion ratios
musVar = [mus./q^muScale for q in qs] # learning steps to be tested for different Ns
SNR = 3 # signal to noise ratio for gaussian noise in the LMS-Like learning rule

# built updates for different learning steps for training 
updatesAll = buildUpdates(musVar,"LMSTrain",[t_train,deltaTe,deltaTr,deltaTh,trajErrorIndex],SNR) 
mu = mus[1]

# build updates to compute the steady state loss
updatesAllSS = buildUpdates(musVar,"LMSTrain",[t_trainSS,deltaTe,deltaTr,deltaTh,trajErrorIndex],SNR) 

# post-process: compute task loss
t_save = t_train
# saveEpochsInt = 5;
# t_save = t_train[2:saveEpochsInt:end]
records = [LearningPerformanceTask(t_save,trainErrorIndex)] # compute task loss

#####
# Prepare paths and files to save variables with DrWatson
#####
saveParams = @strdict plantMatrices Ks fc trajTime lookahead_times num_nn_inputs num_nn_outputs Ns mu deltaTr deltaTe deltaTh t_train simI t_save K records
lblParams = @strdict FB=Ks[1] fc trajTime maxL=lookahead_times[end] mu deltaTr deltaTe deltaTh 
filename = savename(lblParams,"jld2")
path = string(simModel,"_",Ns[1],"_",Ns[end])
mkpath(datadir("simulations", path))

# make plot path
mkpath(plotsdir(path))

#####
# Pretrain network for steady state loss calculation
#####

preTrainType = "GradientTrain"
# preTrainType = "LMSTrain"

if preTrainType=="GradientTrain" # for gradient use larger learning step
    musPT = [0.5] # learning steps
    musVarPT = [mus./q^muScale for q in qs] # learning steps to be tested for different Ns
else 
    musPT = musVar
end

if preTrain
    pathPT = string(simModel,"_",Ns[1],"_",Ns[end],"/preTrain")
    mkpath(datadir("simulations", pathPT))
    wssAll = map(1:length(simI)) do i # for each simulation pretrain with largest learning step and gradient descent
        seed = simI[i]
        mkpath(plotsdir(path,"preTrain_$seed"))
        # for smallest net size 
        lossCutoff=0 # loss cut off for gradient training 
        wss = preTrainF(systemsSim[i],musVarPT,Ns,trajTimePTs,dt,trainErrorIndex,trajTimePP,path, "preTrain_$seed/", preTrainType,lossCutoff) 
    end
    # save pretrain results to reuse if necessary 
    saveParamsPT = @strdict Ns trajTimePTs musVarPT simI wssAll
    fulldPT = copy(lblParams) # set save parameters
    fulldPT["seed1"] = simI[1]
    fulldPT["seedLast"] = simI[end]
    fulldPT["N1"] = Ns[1]
    fulldPT["Nlast"] = Ns[end]
    fulldPT["mu1"] = mus[1]
    fulldPT["muLast"] = mus[end]
    filenamePT = savename(fulldPT,"jld2") 
    wsave(datadir("simulations",pathPT, filenamePT), saveParamsPT)
end

###############
# Simulate to compute learning speed and steady state loss
###############
mm = map(1:length(simI)) do i # for each seed
    seed = simI[i]
    mkpath(plotsdir(path,"train_$seed"))
    s = systemsSim[i] 
    m = map(1:length(s)) do j # for each size
        ss = s[j]
        changeOutputWeights!(ss,wssAll[i][j][1]) # set weights back to initial
        us = updatesAll[j] # updates for this size net
        # train
        m0 = map(1:length(us)) do k 
            u = us[k] 
            t,p=simulate(ss,u,t_save,records,trajTime,trajTimePP,plotsdir(path,"train_$seed/"))
            if verbose # save each result in case simulation crashes
                fulld = copy(saveParams) # set save parameters
                fulld["seed"] = simI[i]
                fulld["N"] = Ns[j]
                fulld["system"] = ss
                fulld["mu"] = musVar[j][k]
                fulld["postS"] = p
                fullLbl=copy(lblParams) # set save label parameters
                fullLbl["seed"] = simI[i]
                fullLbl["N"] = Ns[j]
                fullLbl["mu"] = musVar[j][k]
                filename = savename(fullLbl,"jld2") 
                println(filename)
                wsave(datadir("simulations",path, filename), fulld)
            end
            return t,p
        end
        if preTrain # if pre-train compute ss loss from pre-trained weight state
            mkpath(plotsdir(path,"ss_$seed"))
            t_saveSS = t_trainSS[1]:5:trajTimeSS # save less often for comp efficiency
            recordsSS = [LearningPerformanceTask(t_saveSS,trainErrorIndex)]
            # ss.trajTime = trajTimeSS # change trajectory time to longer
            changeOutputWeights!(ss,wssAll[i][j][2]) # set weights to ss
            m0SS = map(1:length(us)) do k 
                u = us[k] 
                tSS,pSS=simulate(ss,updatesAllSS[j][k],t_saveSS,recordsSS,trajTimeSS,trajTimePP,plotsdir(path,"ss_$seed/")) 
                return m0[k][1],m0[k][2], pSS
            end
            return m0SS
        else
            return m0
        end
    end 
    if ~verbose # save once all nets and mus simulated
        fulld = copy(saveParams) # set save parameters
        fulld["seed"] = simI[i]
        fulld["Ns"] = Ns
        fulld["mu"] = musVar
        fulld["postStores"] = loop_2(m,x->x[2]) # will be of size length(Ns)xlength(mus)
        if preTrain
            fulld["postStoresSS"] = loop_2(m,x->x[3]) # will be of size length(Ns)xlength(mus)
        end
        fullLbl=copy(lblParams) # set save label parameters
        fullLbl["seed"] = simI[i]
        fullLbl["N1"] = Ns[1]
        fullLbl["Nlast"] = Ns[end]
        fullLbl["mu1"] = mus[1]
        fullLbl["muLast"] = mus[end]
        filename = savename(fullLbl,"jld2") 
        println(filename)
        wsave(datadir("simulations",path, filename), fulld)
    end
    return m
end

