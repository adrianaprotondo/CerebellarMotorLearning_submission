#####
# Test learning speed and steady state error from task error 
# consider systems with different activation functions and coding levels

# for different net size Ns and different mus, bound of mus change with net size to make sure good interval
# do NumSimV simulations for each sim, Ns, mu to average out noise from learning 
# Train with LMS each net, starting with same weights but not following same path 
# obtain ss from ss when initial weights are close to min 
# load pretrain weights from file
# save with drWatson the values
#####

include("default.jl")
include("sizeSim_functions.jl")

# simModel = "lms_size_Ls-ss"
simModel = "test_actF_bias_long_2"
# simModelPT = "lms_size_Ls-ss_fourrier_o-u_2_10_90"
# simModelPT = "test_actF_bias_10_70"
simModelPT = false # path name for pretrain values
# simModelPT = "test_actF_bias_long_20_70"

verbose = false # save for every size and every update 
preTrain = true

actF = relu 
varBS = -0.01

if actF==tanh
    actFN="tanh" 
elseif actF==relu 
    actFN ="relu"
end

# numSim = 1;
seed = 12;
simI = [seed]


# Ns = 1*num_nn_inputs:1*num_nn_inputs:10*num_nn_inputs; # number of hidden units in the network to test
# Ns = 1*num_nn_inputs:3*num_nn_inputs:9*num_nn_inputs; # number of hidden units in the network to test
Ns = 2*num_nn_inputs:5*num_nn_inputs:8*num_nn_inputs; # number of hidden units in the network to test
# Ns = 1*num_nn_inputs:2*num_nn_inputs:5*num_nn_inputs; # number of hidden units in the network to test

# trajTimePT = 100.
trajTimePTs = [trajTimePT*(Ns[end]+1-N)^0.1 for N in Ns ]

varBSs = [-1.0,-0.5,-0.1,0.0,0.1,0.5,1.0]

################  
# Define systems
################
# generate reference trajectory as long as the longest trajectory 
maxTime = maximum(vcat(trajTimePTs,trajTime,trajTimeSS,trajTimePP))

Random.seed!(seed)
refF =  refFourier(fc,Nref,"o-u",[1/Nref,2*pi*Nref])

systemsSim = map(varBSs) do var
    build_systems_sim(Ns,plantMatrices,Ks,refF,trajTime,lookahead_times,num_nn_inputs,num_nn_outputs,K,seed;actF=actF,varBS=var);  
end
trainErrorIndex = systemsSim[1][1].trainErrorIndex 


################  
# Train parameters
################
t_trainSS = 0.01:dt:trajTimeSS

# mus = collect(0.0001:0.005:0.08)
# mus = collect(0.01:0.01:0.02)
# mus = collect(0.01:0.01:0.1)
mus = [0.02,0.12]

muScale = 0.5
qs = Ns./Ns[1] 
musVar = [mus./q^muScale for q in qs]
SNR = 3
lmsNR = 0.0

updatesAll = buildUpdates(musVar,"LMSTrain",[t_train,deltaTe,deltaTr,deltaTh,trajErrorIndex],SNR,lmsNR) 
mu = mus[1]

updatesAllSS = buildUpdates(musVar,"LMSTrain",[t_trainSS,deltaTe,deltaTr,deltaTh,trajErrorIndex],SNR,lmsNR) 

t_save = t_train

# records = [LocalTaskDifficulty(-mu,:lms,t_save,trainErrorIndex)]
# records = [LearningPerformanceTask(t_save,trainErrorIndex)]
records = [LearningPerformanceTask(t_save,trainErrorIndex),CodingLevel(t_save,1.0)]

################  
# Saving parameters
################
# values to save 
saveParams = @strdict plantMatrices Ks fc trajTime lookahead_times num_nn_inputs num_nn_outputs Ns mu deltaTr deltaTe deltaTh t_train simI t_save K records varBSs actFN
lblParams = @strdict FB=Ks[1] fc trajTime maxL=lookahead_times[end] mu deltaTr deltaTe deltaTh actFN varBSs 
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

if simModelPT!=false && preTrain # to load pretrain values from simModelPT
    pathPT = string(simModelPT,"/preTrain")
    dfPT = collect_results(datadir("simulations",pathPT);verbose=true) # get data in the path with simulations
    sort!(dfPT,:simI)
    wSS = dfPT[!,:wssAll] 
    # simIs = unique(dfPT[!,:simI]) # sim from 
    varBSsPt = unique(dfPT[!,:varBSs]) # sim from 
    NsPt = dfPT[!,:Ns]
    wssAll = map(varBSs) do s # get weights for each net size for a given seed
        for i=1:length(varBSs)
            if s in varBSsPt[i]
                j=findfirst(x-> x==s, varBSsPt[i])
                w = wSS[i][j]
                # println(w)
                wssAN = map(Ns) do N
                    if N in NsPt[i]
                        println(N)
                        jj = findfirst(x->x==N, NsPt[i])
                        println(w[jj])
                        return w[jj]
                    end
                end
                return wssAN
            end
        end
    end
elseif preTrain # pretrain directly
    pathPT = string(simModel,"_",Ns[1],"_",Ns[end],"/preTrain")
    mkpath(datadir("simulations", pathPT))
    wssAll = map(1:length(varBSs)) do i # for each simulation pretrain with largest learning step and gradient descent
        varBS = varBSs[i]
        mkpath(plotsdir(path,"preTrain_$varBS"))
        # for smallest net size 
        lossCutoff=0 # loss cut off for gradient training 
        wss = preTrainF(systemsSim[i],musVarPT,Ns,trajTimePTs,dt,trainErrorIndex,trajTimePP,path, "preTrain_$varBS/", preTrainType,lossCutoff;stopTraining=false) 
    end
    # save pretrain results to reuse if necessary 
    saveParamsPT = @strdict Ns trajTimePTs musVarPT simI actFN varBSs wssAll
    fulldPT = copy(lblParams) # set save parameters
    fulldPT["seed"] = seed
    fulldPT["varBS1"] = varBSs[1]
    fulldPT["varBSE"] = varBSs[end]
    fulldPT["N1"] = Ns[1]
    fulldPT["Nlast"] = Ns[end]
    fulldPT["mu1"] = mus[1]
    fulldPT["muLast"] = mus[end]
    filenamePT = savename(fulldPT,"jld2") 
    wsave(datadir("simulations",pathPT, filenamePT), saveParamsPT)
end



#############
# Simulate
#############
numSimV=5
numSimVSS=5
# Simulate 
simRes = map(1:length(varBSs)) do i # for each seed
    varBS = varBSs[i]
    mkpath(plotsdir(path,"train_$varBS"))
    s = systemsSim[i]
    m = map(1:length(s)) do j # for each size
        ss = s[j]
        # changeOutputWeights!(ss,wssAll[i][j][1]) # set weights back to initial
        us = updatesAll[j] # updates for this size net
        # train
        m0 = map(1:length(us)) do k # for each mu
            u = us[k]
            map(1:numSimV) do l
                t,p=simulate(ss,u,t_save,records,trajTime,trajTimePP,plotsdir(path,"train_$varBS/"))
                return t,p
            end
        end
        if preTrain
            mkpath(plotsdir(path,"ss_$varBS"))
            t_saveSS = t_trainSS[1]:5:trajTimeSS # save less often for comp efficiency
            recordsSS = [LearningPerformanceTask(t_saveSS,trainErrorIndex)]
            changeOutputWeights!(ss,wssAll[i][j][2]) # set weights to ss
            m0SS = map(1:length(us)) do k 
                u = us[k] 
                map(1:numSimVSS) do l
                    tSS,pSS=simulate(ss,updatesAllSS[j][k],t_saveSS,recordsSS,trajTimeSS,trajTimePP,plotsdir(path,"ss_$varBS/")) 
                    return m0[k][l][1],m0[k][l][2], pSS
                end
            end
            return m0SS
        else
            return m0
        end
    end 
    if ~verbose # save once all nets and mus simulated
        fulld = copy(saveParams) # set save parameters
        fulld["seed"] = seed
        fulld["varBS"] = varBSs[i]
        fulld["Ns"] = Ns
        fulld["numSimV"] = numSimV
        fulld["numSimVSS"] = numSimVSS
        fulld["mu"] = musVar
        # fulld["updates"] = updatesAll
        # fulld["trainStores"] = loop_2(m,x->x[1]) # save all stores for each size, each update
        fulld["postStores"] = loop_3(m,x->x[2]) # will be of size length(Ns)xlength(mus)
        if preTrain
            fulld["postStoresSS"] = loop_3(m,x->x[3]) # will be of size length(Ns)xlength(mus)
        end
        fullLbl=copy(lblParams) # set save label parameters
        fullLbl["seed"] = seed
        fullLbl["varBS"] = varBSs[i]
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

