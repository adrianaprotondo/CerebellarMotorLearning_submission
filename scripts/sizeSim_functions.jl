#######
# Helper functions for simulations of motor control system: 
## functions to build expanded systems with size N given original network with some parameters
## function build_systems_sim returns systems with different net sizes given by Ns with random seed randomSeed
## function build_systems_sim_K: build all the systems with sizes Ns and input sparsity Kss but same ref, plant... keep same W for different K. expand W with zeros for different N. returns vector of vector of systems with length(Kss)xlength(Ns)
## function build_systems_sim_K_KNconst: same as above but keeping the number of input connections to Kss[end]*N (i.e. to the largest size)
## plotting functions
## helper functions
######
using DataFrames
using StatsPlots

###############
# Build systems 
###############
"""
build system of size N, and K indegree given parameters and matrices of smallest net Z0 and W0
params = [plant,pid,nnDims,ref,trajTime,lookahead_times,Z0,W0] 
"""
function build_system_N(params,N,K,outputIndex=3;actF=tanh,varBS=0.0)
    params = systemParams!(params,N,K)
    biasV = varBS/K
    build_system(params...,outputIndex;actF=actF,varB=biasV)
end

"""
build inputsystem with K inputs per gc and N GCs given parameters and matrices of smallest net Z0 and W0
params = [nnDims,ref,trajTime,lookahead_times,Z0,W0] 
"""
function build_inputSystem_N(params,N,K;actF=tanh,varBS=0.0)
    params = systemParams!(params,N,K,1)
    biasV = varBS/K
    build_inputSystem(params...;actF=actF,varB=biasV)
end

"""
    build parameters for system of size N,and K indegree 
    given parameters and matrices of smallest net Z0 and W0
    dimsI is the index of NNdims 
    params = [plant,pid,nnDims,ref,trajTime,lookahead_times,Z0,W0] 
    or 
    params = [nnDims,ref,trajTime,lookahead_times,Z0,W0] 
"""
function systemParams!(params,N,K,dimsI=3)
    Z0 = params[end-1]
    W0 = params[end]
    nnDims = params[dimsI]
    K0 = size(Z0,2)-sum(Z0[1,:].==0)
    if K0!=K # if K is changed
        Z = createZ(nnDims[1],N,K) # change input matrix with new
        params[end-1] = Z
    elseif size(Z0,1) < N # if K is not changed but expansion
        Z = createZ(nnDims[1],N,K)
        Z[1:size(Z0,1),:] = Z0
        params[end-1] = Z
    end
    if size(W0,2) < N # if net is larger than weight matrices
        W = zeros(nnDims[3],N)
        W[:,1:size(W0,2)] = W0
        params[dimsI] = (nnDims[1],N,nnDims[3])
        params[end] = W 
    end
    return params
end

function nn_matrices(numDims,K)
    num_nn_inputs, N, num_nn_outputs = numDims
    Z0 = createZ(num_nn_inputs,N,K)
    d = Normal(0,1/sqrt(N))
    W0 = rand(d,(num_nn_outputs,N))
    return Z0, W0
end 

"""
    build systems for simulation randomSeed sets seed for weight matrices and reference 
    build all the systems with sizes Ns but same ref, plant... 
"""
function build_systems_sim(Ns,plantMatrices,Ks,refF,trajTime,lookahead_times,num_nn_inputs,num_nn_outputs,K,randomSeed;actF=tanh,varBS=0.0) 
    Random.seed!(randomSeed)
    nnDims = (num_nn_inputs,Ns[1],num_nn_outputs)
    Z0, W0 = nn_matrices(nnDims,K)

    biasV = varBS/K
    system = build_system(plantMatrices,Ks,nnDims,refF,trajTime,lookahead_times,Z0,W0;actF=actF,varB=biasV);
    plant = system.plant
    pid = system.pid
    ref = system.ref
    # reuse plant, pid and ref
    paramsN = [plant,pid,nnDims,ref,trajTime,lookahead_times,Z0,W0]
    systems = map(Ns) do N 
        build_system_N(paramsN,N,K;actF=actF,varBS=varBS)
    end
    return systems
end

function build_systems_sim(Ns,plant::Plant,Ks,refF,trajTime,lookahead_times,num_nn_inputs,num_nn_outputs,K,randomSeed,outputIndex;actF=tanh,varBS=0.0) 
    Random.seed!(randomSeed)
    nnDims = (num_nn_inputs,Ns[1],num_nn_outputs)
    Z0, W0 = nn_matrices(nnDims,K)

    biasV = varBS/K
    system = build_system(plant,Ks,nnDims,refF,trajTime,lookahead_times,Z0,W0,outputIndex;actF=actF,varB=biasV);
    pid = system.pid
    ref = system.ref

    # reuse plant, pid and ref
    paramsN = [plant,pid,nnDims,ref,trajTime,lookahead_times,Z0,W0]
    systems = map(Ns) do N 
        build_system_N(paramsN,N,K,outputIndex;actF=actF,varBS=varBS)
    end
    return systems
end

"""
    build systems for simulation randomSeed sets seed for weight matrices and reference 
    build all the systems with sizes Ns and input sparsity Kss but same ref, plant...
    keep same W for different K 
    expand W with zeros for different N
    return vector of vector of systems with length(Kss)xlength(Ns) 
"""
function build_systems_sim_K(Ns,plantMatrices,Ks,refF,trajTime,lookahead_times,num_nn_inputs,num_nn_outputs,Kss,randomSeed;actF=tanh,varBS=0.0) 
    Random.seed!(randomSeed)
    nnDims = (num_nn_inputs,Ns[1],num_nn_outputs)
    Z0 = createZ(num_nn_inputs,Ns[1],Kss[1])
    d = Normal(0,1/sqrt(Ns[1]))
    W0 = rand(d,(num_nn_outputs,Ns[1]))

    biasV = varBS/K
    system = build_system(plantMatrices,Ks,nnDims,refF,trajTime,lookahead_times,Z0,W0;actF=actF,varB=biasV);
    plant = system.plant
    pid = system.pid
    ref = system.ref
    # reuse plant, pid and ref
    # paramsN = [plant,pid,nnDims,ref,trajTime,lookahead_times,Z0,W0]
    systems = map(Kss) do K
        Z0 = createZ(num_nn_inputs,Ns[1],K) # new input weight matrix for K
        map(Ns) do N 
            build_system_N([plant,pid,nnDims,ref,trajTime,lookahead_times,Z0,W0],N,K;actF=actF,varBS=varBS)
        end
    end
    return systems
end

"""
    build systems for simulation randomSeed sets seed for weight matrices and reference 
    build all the systems with sizes Ns and input sparsity Kss but same ref, plant...
    keep number of input connections to Kss[end]*N
    keep same W for different K 
    expand W with zeros for different N
    return vector of vector of systems with length(Kss)xlength(Ns) 
"""
function build_systems_sim_K_KNconst(N,plantMatrices,Ks,refF,trajTime,lookahead_times,num_nn_inputs,num_nn_outputs,Kss,randomSeed;actF=tanh,varBS=0.0) 
    Random.seed!(randomSeed)
    Ns = trunc.(Int,(Kss[end]*N)./Kss)# keep the number of input connections constant to K[end]*N
    nnDims = (num_nn_inputs,Ns[end],num_nn_outputs)
    Z0 = createZ(num_nn_inputs,Ns[end],Kss[end])
    d = Normal(0,1/sqrt(Ns[end]))
    W0 = rand(d,(num_nn_outputs,Ns[end]))

    biasV = varBS/Kss[end]
    system = build_system(plantMatrices,Ks,nnDims,refF,trajTime,lookahead_times,Z0,W0;actF=actF,varB=biasV);
    plant = system.plant
    pid = system.pid
    ref = system.ref
    # reuse plant, pid and ref
    # paramsN = [plant,pid,nnDims,ref,trajTime,lookahead_times,Z0,W0]
    systems = map(1:length(Kss)) do i
        # Z0 = createZ(num_nn_inputs,Ns[i],Kss[i]) # new input weight matrix for K
        build_system_N([plant,pid,nnDims,ref,trajTime,lookahead_times,Z0,W0],Ns[i],Kss[i];actF=actF,varBS=varBS)
    end
    return systems
end


#####
# Input systems 
#####
"""
    build input systems for simulation randomSeed sets seed for weight matrices and reference 
    build all the inputsystems with sizes Ns but same ref 
    actF is the activation function for the NN 
    varBS is the scaling of the variance of the bias
    will be set to varBS/K
"""
function build_systems_sim(Ns,refF,trajTime,lookahead_times,num_nn_inputs,num_nn_outputs,K,randomSeed;actF=tanh,varBS=0.0) 
    Random.seed!(randomSeed)
    nnDims = (num_nn_inputs,Ns[1],num_nn_outputs)
    Z0, W0 = nn_matrices(nnDims,K)

    biasV = varBS*1/K # spread of the biases depends on number of inputs to each unit
    # biasV = varBS/sqrt(K)
    system = build_inputSystem(nnDims,refF,trajTime,lookahead_times,Z0,W0;actF=actF,varB=biasV);
    ref = system.ref
    # reuse plant, pid and ref
    # paramsN = [nnDims,ref,trajTime,lookahead_times,Z0,W0]
    systems = map(Ns) do N 
        build_inputSystem_N([nnDims,ref,trajTime,lookahead_times,Z0,W0],N,K;actF=actF,varBS=varBS)
    end
    return systems
end

"""
    build input systems for simulation randomSeed sets seed for weight matrices and reference 
    build all the inputsystems with size N and Ks but same ref 
    if constantNumWeights keep the number of weights N*K constant, else keep N constant for different K
    actF is the activation function for the NN 
    varBS is the scaling of the variance of the bias
        will be set to varBS/K
"""
function build_systems_sim_K(N,refF,trajTime,lookahead_times,num_nn_inputs,num_nn_outputs,Ks,randomSeed,constantNumWeights=false;actF=tanh,varBS=0.0) 
    Random.seed!(randomSeed)
    if constantNumWeights # if keep the number of input connections constant to K[end]*N
        Ns = trunc.(Int,(Ks[end]*N)./Ks)
    else # if keep the number of granule cells constant
        Ns = trunc.(Int,N.*ones(length(Ks)))
    end
    # reference values for the smallest number of granule cells (Ns[end] and Ks[end])
    nnDims = (num_nn_inputs,Ns[end],num_nn_outputs) 
    Z0, W0 = nn_matrices(nnDims,Ks[end])

    biasV = varBS*1/Ks[end] # spread of the biases depends on number of inputs to each unit
    system = build_inputSystem(nnDims,refF,trajTime,lookahead_times,Z0,W0;actF=actF,varB=biasV);
    ref = system.ref
    # reuse plant, pid and ref
    # paramsN = [nnDims,ref,trajTime,lookahead_times,Z0,W0]
    systems = map(1:length(Ks)) do i
        build_inputSystem_N([nnDims,ref,trajTime,lookahead_times,Z0,W0],Ns[i],Ks[i];actF=actF,varBS=varBS)
    end
    return systems
end

"""
    Function that returns output weights of an expanded neural network with N weights
    based on the weight matrix W0 of smaller net 
"""
function expandW(W0,N)
    W = zeros(N,size(W0,2))
    W[1:size(W0,1),:] = W0
    return W 
end

""" Change ouptut weights of the system's nn 
    W should be of size 1xN1
    new weights of system ss fluxnet are set to [W, 0,0,..,0]
    i.e. the passed weights padded with zeros at the end if the 
    number of weights in W and the net of the system is mismatched
"""
function changeOutputWeights!(system,W)
    s = system.nn.fluxNet
    Wss = zeros(1,system.nn.dims[2])
    Wss[1,1:size(W,2)] = W
    Flux.params(s)[3] .= Wss
    system
end 

""" Pretrain an array of systems with different sizes. Return final weights and initial weights for each one"""
function preTrainF(systems,musVar,Ns,trajTimePTs,dt,trainErrorIndex,trajTimePP,path,plotsPath,funcType="GradientTrain",lossCutoff=0,ssWI=10,lmsParam=[1.0,0.5,0.1],trajErrorIndex=2;stopTraining=true)
    wss=map(1:length(Ns)) do j # for each size
        t_trainPT = 0.01:dt:trajTimePTs[j] # train times for pretraining 
        # t_savePT = t_trainPT[1]:50:trajTimePTs[j] # save less often for comp efficiency
        int = maximum([trajTimePTs[j]/30,1])
        t_savePT = t_trainPT[1]:int:trajTimePTs[j] # save less often for comp efficiency
        recordsPT = [LearningPerformanceTask(t_savePT,trainErrorIndex)] # get task loss of pre-training 
        muPT = musVar[j][end] # use largest learning step
        if funcType=="GradientTrain" # pre-train with gradient descent
            uPT = [GradientTrain(muPT,t_trainPT,trainErrorIndex,trajTimePP,lossCutoff)];
        elseif funcType=="LMSTrain" # pre-train with LMS learning rule
            deltaTePT,deltaTrPT,deltaTrPT = lmsParam
            uPT = [LMSTrain(muPT,t_trainPT,deltaTePT,deltaTrPT,deltaTrPT,trajErrorIndex)];
        end
        s1 = systems[j] # train all nets
        t,p = simulate(s1,uPT,t_savePT,recordsPT,trajTimePTs[j],trajTimePP,plotsdir(path,plotsPath);computeCL=false)
        wInitial = t[:weights][:,1]' # store initial weights to set after pre-training to maintain initialisation
        wss = mean(t[:weights][:,end-ssWI:end]';dims=1) # ss weights are mean over interval 
        changeOutputWeights!(s1,wInitial) # change output weights back to initial for training 
        if stopTraining && p[:taskError][1,end]>lossCutoff # if the task loss at end of pretrain is larger than cutoff 
            lossCutoff=p[:taskError][1,end] # set as new cutoff for next net sizes this assures that nets achieve approximate same pre-train
        end
        return wInitial, wss
    end
    return wss
end

""" build array of array of updates for each learning step in musVar
    trainType is the function to create updates 
        arg is the arguments for trainType after the learning step should be [t_train,deltaTe,deltaTr,deltaTh,trajErrorIndex]
    SNR if larger than zero add gaussian noise with gamma = learning step/SNR 
    lmsNR if larger than zero add noise inside lms
"""
function buildUpdates(musVar,trainType,arg,SNR=0,lmsNR=0)
    mm = map(musVar) do d # all sizes
        map(d) do dd # all learning steps 
            if trainType == "LMSTrain"
                if lmsNR>0 # add noise to LMS
                    a = LMSTrain(dd,arg...,lmsNR) # training function  
                else
                    a = LMSTrain(dd,arg...) # training function 
                end
            elseif trainType == "OnlineGradientTrain"
                a = OnlineGradientTrain(dd,arg...); 
            end
            if SNR>0 # add gaussian noise
                gamma=dd/SNR
                b = GaussianNoise(gamma,arg[1]); # arg[1] should be trainTime
                m = [a,b]
            else
                m = [a]
            end
            m
        end
    end
    return mm
end

###############
# Save variables
###############
"""
    create file to save variables return filename generated from parameters
    save the parameters given into the file
"""
function createSaveFile(saveParams,filePath="../Variables/")
    mu = saveParams["mu = "]
    gamma = saveParams["gamma = "]
    Ns = saveParams["Ns = "]
    deltaTe = saveParams["deltaTe = "]
    fileName = string(filePath,"sizeTest_mu-",mu,"_gamma-",gamma,"_deltaTe-",deltaTe,"_Ns-",Ns[1],"-",Ns[end],".jl")
    output_file = open(fileName,"w")
    write(output_file,"# Parameters \n \n")
    for (key, value) in saveParams
        write(output_file,key)
        show(output_file,value)
        write(output_file, "; \n \n")
    end 
    close(output_file)
    return fileName
end

"""
    save variable tl with name in fileName
"""
function saveVar(fileName,name,tl)
    output_file = open(fileName,"a")
    write(output_file,name)
    show(output_file,tl)
    write(output_file, "; \n \n") 
    close(output_file)
end

###############
# Analyse simulation results
###############
""" 
loop over a vector of vectors with five layers and apply the function fc to each element
"""
function loop_5(trainStores,fc,args=[])
    map(trainStores) do t
        map(t) do tt
            map(tt) do ttt
                map(ttt) do tttt
                    map(tttt) do ttttt
                        fc(ttttt,args...)
                    end
                end
            end
        end
    end
end

""" 
loop over a vector of vectors with four layers and apply the function fc to each element
"""
function loop_4(trainStores,fc,args=[])
    map(trainStores) do t
        map(t) do tt
            map(tt) do ttt
                map(ttt) do tttt
                    fc(tttt,args...)
                end
            end
        end
    end
end

""" 
loop over a vector of vectors with three layers and apply the function fc to each element
"""
function loop_3(trainStores,fc,args=[])
    map(trainStores) do t
        map(t) do tt
            map(tt) do ttt
                fc(ttt,args...)
            end
        end
    end
end

""" 
loop over a vector of vectors with two layers and apply the function fc to each element
"""
function loop_2(trainStores,fc,args=[])
    map(trainStores) do t
        map(t) do ttt
            fc(ttt,args...)
        end
    end
end

""" 
loop over a vector of vectors with two layers and apply the function fc to each element
"""
function loop(trainStores,fc,args=[])
    map(trainStores) do t
        fc(t,args...)
    end
end

""" 
normalize p by p[1] if non_zero
"""
function normalize_by1(p)
    if p[1]>0.0
        return p./p[1]
    else
        return p 
    end
end 

"""
    extract train stores and post stores of simulation result mm 
    when simulating over two parameters (size and initialisation for example)
"""
function simAnsAnalisis_2(mm)
    trainStores = loop_2(mm,x -> x[1]) 
    postStores = loop_2(mm,x -> x[2]) 
    return trainStores,postStores
end 

"""
    extract trainstores and poststores from solution running simulations with three parameters changing
"""
function simAnsAnalisis_3(mm)
    trainStores = loop_3(mm,x -> x[1]) 
    postStores = loop_3(mm,x -> x[2]) 
    return trainStores,postStores
end 


"""
    extract train stores and post stores of simulation result mm 
    and extract learning speed and steady state error from postStores
    when simulating over two parameters (size and initialisation for example)
"""
function simAnsAnalyse(mm)
    trainStores, postStores = simAnsAnalisis_2(mm)
    ls = postStoresExtract_2(postStores,:learningSpeed)
    lsNorm = map(ls) do p
        normalize_by1(p)
    end
    ss = postStoresExtract_2(postStores,:steadyStateE) 
    ssNorm = map(ss) do p
        normalize_by1(p)
    end
    return trainStores,postStores,ls,ss, lsNorm,ssNorm
end


"""
    extract train stores and post stores of simulation result mm 
    and extract learning speed and steady state error from postStores
    when simulating over three parameters (size, initialisation and deltaTe for example)
"""
function simAnsAnalyse_3(mm)
    trainStores, postStores = simAnsAnalisis_3(mm)
    ls = postStoresExtract_3(postStores,:learningSpeed)
    lsNorm = loop_2(ls,normalize_by1)
    ss = postStoresExtract_3(postStores,:steadyStateE) 
    ssNorm = loop_2(ss,normalize_by1)
    return trainStores,postStores,ls,ss,lsNorm,ssNorm
end

"""
    extract from an array of postStores the variable with symbol symb
    array of postores has three levels
"""
function postStoresExtract_3(postStores,symb)
    map(postStores) do p
        postStoresExtract_2(p,symb)
    end
end
"""
    extract from an array of postStores the variable with symbol symb
    array of postores has two levels
"""
function postStoresExtract_2(postStores,symb)
    map(postStores) do p
        _postStoresExtract(p,symb)
        # map(p) do pp
        #     pp[symb]
        # end
    end
end

"""
    extract from an array of postStores the variable with symbol symb
    array of postores has one level
"""
function _postStoresExtract(postStores,symb)
    map(postStores) do p
        p[symb]
    end
end
# """
#     extract from an array of postStores the variable with symbol symb
#     array of postores has one level
# """
# function postStoresExtract(postStores,symb)
#     p = postStores
#     while typeof
#     map(postStores) do p
#         p[symb]
#     end
# end

"""
    get run function f on postStore[s]
"""
function getValF(postStore,s,f)
    return f(postStore[s])
    # dF = postStore[Symbol(:localTaskDifficulty_,s)]
    # return findmin(dF)[1]
end

"""
    get run function f on postStore[int][s]
"""
function getValF(postStore,s,f,int)
    return f(postStore[int][s])
end

"""
    return array of arrays of arrays with 
    elements from int selected in level 3
"""
function selectInd_3(p,int)
    map(1:length(p)) do i 
        map(int) do j
            map(1:length(p[i])) do k 
                p[i][k][j]
            end
        end
    end
end

""" 
    decompose vec into the direction of ref plus noise
    hat{vec} = γ₁hat{ref} + γ₂hat{n}
    return gamma1, gamma2, and n
"""
function decompose(vec,ref)
    gamma1 = zeros(1,size(vec,2))
    gamma2 = zeros(1,size(vec,2))
    n = zeros(size(vec))
    for i=1:size(vec,2)
        if norm(vec[:,i])>0
            nv = normalize(vec[:,i])
            nr = normalize(ref[:,i])
            gamma1[:,i] .= (nv'*nr)
            noise = nv.-gamma1[:,i].*nr
            gamma2[:,i] .= norm(noise)
            n[:,i] .= normalize(noise)
        end
    end
    return gamma1,gamma2,n 
end


"""
    reuturn norm of each slice of value of trainstore at symb
"""
function calculateNorm(trainStore,symb=:dw)
    mapslices(norm,trainStore[symb],dims=1)'
end


###############
# Plotting functions
###############
"""
    compute standard error of an array of arrays
"""
function my_std(ls)
    numSim = length(ls)
    lsM = hcat(ls...)
    map(1:size(lsM,1)) do m
        std(lsM[m,:])./sqrt(numSim)
    end
end

"""
    plot mean of ls vs Ns 
    label of each line is given by l 
    the xlabel is given by xl and ylabel by yl
    save figure with name saveLbl
"""
function plot_mean(Ns,ls,l,xl,yl,saveLbl,vertical=false,lt=:scatter,cm=false,ms=5)
    if cm==false
        my_colors = ["#D43F3AFF", "#5CB85CFF", "#EEA236FF", "#46B8DAFF","#357EBDFF", "#9632B8FF", "#B8B8B8FF"]
    else
        my_colors = cm
    end
    lsm = mean(ls)
    lss = my_std(ls)
    if length(l) == 1 && length(lsm) > 1# if only one line to plot
        if lt==:scatter
            plot(Ns,lsm,yerr=lss,lw=3,label=l[1],color=my_colors[1],seriestype=lt,legend=:outertopright,grid=false,markersize=ms,ylim=(0.0,Inf))
        elseif ls==:line
            plot(Ns,lsm,ribbon=lss,lw=3,label=l[1],color=my_colors[1],seriestype=lt,legend=:outertopright,grid=false,markersize=ms)
        end
        if (vertical==argmax) || (vertical==argmin)
            vline!([Ns[vertical(lsm)]],lw=2,ls=:dash,color=my_colors[1],label = string("optimum ", l[1]))
        end 
    else
        if lt==:scatter
            plot(Ns,lsm[1],yerr=lss[1],lw=3,label=l[1],color=my_colors[1],seriestype=lt,legend=:outertopright,grid=false,markersize=ms)
        elseif lt==:line
            plot(Ns,lsm[1],ribbon=lss[1],lw=3,label=l[1],color=my_colors[1],seriestype=lt,legend=:outertopright,grid=false,markersize=ms)
        end
        if (vertical==argmax) || (vertical==argmin)
            vline!([Ns[vertical(lsm[1])]],lw=2,color=my_colors[1],ls=:dash,label = string("optimum ", l[1]))
        end
        if length(lsm)>1
            for i=2:length(lsm)
                if lt==:scatter
                    plot!(Ns,lsm[i],yerr=lss[i],lw=3,color=my_colors[i],seriestype=lt,label=l[i],markersize=ms)
                elseif lt==:line
                    plot!(Ns,lsm[i],ribbon=lss[i],lw=3,color=my_colors[i],seriestype=lt,label=l[i],markersize=ms)
                end
                if vertical==argmax || vertical==argmin
                    vline!([Ns[vertical(lsm[i])]],lw=2,color=my_colors[i],ls=:dash,label = string("optimum ", l[i]))
                end
            end
        end
    end
    plot!(xlabel=xl)
    plot!(ylabel=yl)
    savefig(saveLbl)
end

"""
    plot scatter of mean of ss and mean of ls at interval int
    assume ss and ls have shape numSim x length(Ns) x length(t_save)
    calculate mean over numSim for each Ns at t_save[int]
"""
function plot_scatterMean(ss,ls,l,int=false,saveLbl="../Figures/deltaTe/ssVsls_$int.pdf",cm=false,ms=4,stdErr=false,xlbl="local task difficulty")
    if int==false # take all the gammas
        ssS = ss
        lsS = ls
    else # select and interval of gammas
        ssS = map(1:length(ss)) do i
            map(1:length(ss[i])) do j
                ss[i][j][int]
            end
        end
        lsS = map(1:length(ls)) do i
            map(1:length(ls[i])) do j
                ls[i][j][int]
            end
        end
    end
    ssM = mean(ssS)
    lsM = mean(lsS)
    if cm == false
        namedColors = ["Blues","Greens","Oranges","Purples","Grays","Reds"]
    else 
        namedColors = cm
    end
    if stdErr ==true && length(ss)>1
        ssS = my_std(ssS)
        lsS = my_std(lsS)
        scatter(ssM[1],lsM[1],c=colormap(namedColors[1],length(ssM[1])+2)[3:end],label=l[1],colorbar=true,grid=false,markersize = ms,yerr=lsS[1],xerr=ssS[1])
        # legend=:outertopright,
    else
        scatter(ssM[1],lsM[1],c=colormap(namedColors[1],length(ssM[1])+2)[3:end],label=l[1],colorbar=true,grid=false,markersize = ms)
    end
    if length(lsM)>1
        for i=2:length(lsM)
            # scatter!(ssM[i],lsM[i],c=colormap(namedColors[i],length(ssM[i])+2)[3:end],label=l[i],markersize = ms)
            if stdErr == true && length(ss)>1
                scatter!(ssM[i],lsM[i],c=colormap(namedColors[i],length(ssM[i])+2)[3:end],label=l[i],markersize = ms,yerr=lsS[1],xerr=ssS[1])
            else
                scatter!(ssM[i],lsM[i],c=colormap(namedColors[i],length(ssM[i])+2)[3:end],label=l[i],markersize = ms)
            end
        end 
    end
    plot!(xlabel=xlbl)
    plot!(ylabel="learning speed")
    savefig(saveLbl)
end

"""
    Plot mean of lsAll at indeces int vs mus for each N
    lsAll has shape length(numSim)xlength(Ns)xlenth(mus)
    lbl has label for each different line
"""
function plotLines(lsAll,int,mus,lbl,xlbl,ylbl,svlbl,cm)
    lsAllM = hcat(mean(lsAll[int])...)
    lsAllStd = hcat(my_std(lsAll[int])...)
   
    plot(mus,lsAllM,lw=3,label=lbl,legend=:outertopright,
        xlabel=xlbl,ylabel=ylbl,palette=cm,
        yerr = lsAllStd)
    savefig(plotsdir(path,svlbl))
end

##########
# Plotting functions for dataframes 
##########
"""
    Plot ls vs mu and ss vs mu (with lines for each simulation) for each size
    returns plot with 2 columns and as many rows as different sizes
"""
function plotPerSize(sizeIs,Ns,dfAllE,xInd,lsInd,ssInd,lbl,savePath,lsStd=false,ssStd=false,xlbl="learning step",ylbl="learning speed",ylbl2="steady state loss"; groupBy=1, axScale=:none )
    println(groupBy)
    p1s = map(sizeIs) do i
        Ni = Ns[i]
        dfi = filter(:N => ==(Ni),dfAllE)
        if lsStd==false 
            p1 = @df dfi plot(cols(xInd),cols(lsInd),group=cols(groupBy),xlabel=xlbl, ylabel=ylbl,yaxis=axScale) 
            p2 = @df dfi plot(cols(xInd),cols(ssInd),group=cols(groupBy), xlabel=xlbl, ylabel=ylbl2,legend=false,yaxis=axScale)
        else
            p1 = @df dfi plot(cols(xInd),cols(lsInd),group=cols(groupBy) ,yerr=cols(lsStd),xlabel=xlbl, ylabel=ylbl,yaxis=axScale) 
            p2 = @df dfi plot(cols(xInd),cols(ssInd),group= cols(groupBy) ,yerr=cols(ssStd), xlabel=xlbl, ylabel=ylbl2,legend=false,yaxis=axScale)
        end
        plot!(bottom_margin = 10mm)
        # plot(p1,p2,layout=(1,2),size=(pwidth,pheight),title=lbl[i])
        plot(p1,p2,layout=(1,2),title=lbl[i])
        plot!(left_margin=10mm)
    end
    p=plot(p1s...,layout=(:,1),size=(pwidth,(pheight/1.5)*length(sizeIs)))
    savefig(savePath)
    return p
end



"""
    plot var in col(yInd) versus var in col(xInd) (lines for each size)
    for different simulations 
    2 columns and half as many rows as simulations
    dont recomend more than 8 different simulations
"""
function plotPerSeed(seeds,sizeIs,Ns,dfAllE,xInd,yInd,lbl,savePath,xlbl="learning step",ylbl="learning speed",yStd=false;sSymb=:seed,tLbl="seed")
    ps1 = map(seeds) do s
        p1 = plot()
        for j in 1:length(sizeIs)
            i = sizeIs[j]
            dfi = filter(sSymb=> ==(s),filter(:N => ==(Ns[i]),dfAllE))
            if yStd==false
                @df dfi plot!(cols(xInd),cols(yInd),label=lbl[i])
            else
                @df dfi plot!(cols(xInd),cols(yInd),yerr=cols(yStd),label=lbl[i])
            end
        end
        plot!(xlabel=xlbl,left_margin=10mm)
        plot!(ylabel=ylbl,title="$tLbl=$s")
        return p1
    end
    p=plot(ps1..., layout = (:,2),size=(pwidth,pheight*length(seeds)/2))
    savefig(savePath)
    return p
end


@. modelLC(x,p) = p[1]*x+p[2]
"""
    scatter for all Ns in sizeIs 
        values in column with xInd vs values in col yInd 
        if xStd adn ystd plot those error bars 
        plot linear fit for each sizeIs
"""
function scatterCloud(sizeIs,Ns,dfAllE,xInd,yInd,lbl,savePath,xlbl="steady state loss",ylbl="learning speed",namedColors=false,xStd=false,yStd=false;ms=6,axScale=:none,filterBy=:N,trend=true,yaxScale=:none)
    if namedColors==false 
        namedColors = ["Blues","Greens","Oranges","Purples","Reds","Grays"]
        # namedColors = [:Blues,:Greens,:Oranges,:Purples,:Reds,:Grays]
    end
    p = scatter()
    for j in 1:length(sizeIs)
        i = sizeIs[j]
        dfi = filter(filterBy => ==(Ns[i]),dfAllE)
        sort(dfi,:mu)
        p0 =[0.1,minimum(dfi[!,yInd])]
        gd = groupby(dfi,:mu)
        dfM = combine(gd,[yInd=>mean=>:ymean,xInd=>mean=>:xmean])
        fit = curve_fit(modelLC, dfM[!,:xmean], dfM[!,:ymean],p0) # fit the data for trend line
        # fit = curve_fit(modelLC, dfi[!,xInd], dfi[!,yInd],p0) # fit the data for trend line
        colM = colormap(namedColors[j],length(dfi[!,xInd])+3)
        # colM = cgrad(namedColors[j])
        if xStd==false
            # @df sort(dfi,:mu) scatter!(cols(xInd),cols(yInd),label=lbl[i],c=colM[3:end-1],markersize = ms)
            @df sort(dfi,:mu) scatter!(cols(xInd),cols(yInd),label=lbl[i],c=colM[3:end-1],markersize = ms,xaxis=axScale,yaxis=yaxScale)
            # @df sort(dfi,:mu) scatter!(cols(xInd),cols(yInd),marker_z=:mu,label=lbl[i],color=colM,markersize = ms,colorbar=true)
        else
            @df sort(dfi,:mu) scatter!(cols(xInd),cols(yInd),xerr=cols(xStd),yerr=cols(yStd),label=lbl[i],c=colM[3:end-1],markersize = ms,xaxis=axScale,yaxis=yaxScale)
            # @df sort(dfi,:mu) scatter!(cols(xInd),cols(yInd),marker_z=:mu,xerr=cols(xStd),yerr=cols(yStd),label=lbl[i],c=colM,markersize = ms)
        end
        # @df dfM plot!(:xmean,:ymean)
        # @df sort(dfi,:mu) plot!(cols(xInd),modelL(cols(xInd),fit.param),label=lbl[i],color=colM[Int(round(length(colM)/2))])
        if trend
            x = range(minimum(dfM[!,:xmean]),maximum(dfM[!,:xmean]),length=100)
            plot!(x,modelL(x,fit.param),label=lbl[i],color=colM[Int(round(length(colM)/2))])
        end
    end
    plot!(xlabel=xlbl)
    plot!(ylabel=ylbl)
    savefig(savePath)
    return p
end

"""
scatter of var in col(yInd) versus var in col(xInd) (lines for each size)
for different simulations 
2 columns and half as many rows as simulations
dont recomend more than 8 different simulations
"""
function scatterPerSeed(seeds,sizeIs,Ns,dfAllE,xInd,yInd,savePath,xlbl="learning step",ylbl="learning speed",namedColors=false,ms=8)
    if namedColors==false 
        namedColors = ["Blues","Greens","Oranges","Purples","Reds","Grays"]
    end
    ps1 = map(seeds) do s
        p1 = plot()
        for j in 1:length(sizeIs)
            i = sizeIs[j]
            dfi = filter(:seed=> ==(s),filter(:N => ==(Ns[i]),dfAllE))
            colM = colormap(namedColors[j],length(dfi[!,xInd])+3)
            @df sort(dfi,:mu) scatter!(cols(xInd),cols(yInd),label=lbl[i],c=colM[3:end-1],markersize = ms)
        end
        plot!(xlabel=xlbl,ylabel=ylbl)
        plot!(left_margin=10mm,title="seed = $s")
        return p1
    end
    p=plot(ps1..., layout = (:,2),size=(pwidth,pheight*length(seeds)/4))
    plot!(xlabel=xlbl,ylabel=ylbl)
    savefig(savePath)
    return p
end


"""
    for each seed
    plot or scatter col xind of dfE vs col yind 
    all points have the same color
    plot fit with fitT 
    filter out nan values
"""
function plot_qVsVarAllt(dfE,seeds,xInd,yInd,xlbl,ylbl,lblS,savePath,namedColors=false,fitT=false,scatter=true,ms=6,groupProp=[:q,:N,:mu,:seed],stdI=false)
    if namedColors==false 
        namedColors = ["Blue","Green","Orange","Purple","Red","Gray"]
    end
    pl = plot()
    for i=eachindex(seeds)
        dfi = filter(yInd => x -> !isnan(x) , filter(:seed => ==(seeds[i]),dfE))
        if fitT==modelE # fit the mean otherwise stackoverflow
            p0 = [0.8,1.0]
            try 
                fit = curve_fit(fitT,dfi[!,xInd],dfi[!,yInd],p0,upper=[1.2,10],lower=[0.0,0.0])
            catch 
                println("fit mean")
                gd = groupby(dfi,groupProp)
                dfM = combine(gd,[yInd=>mean=>:mean,yInd=>std=>:std])
                x = dfM[!,:q]
                # println(x)
                y = dfM[!,:mean]
                # println(y)
                fit = curve_fit(fitT,x,y,p0,upper=[1.2,10],lower=[0.0,0.0])
            end
            # fit = power_fit(dfi[!,xInd],dfi[!,yInd])
            # plot!(x,y,lw = 4,label=lblS[i],yerr=dfM[!,:std])
        else 
            p0 = [0.1]
            fit = curve_fit(fitT,dfi[!,xInd],dfi[!,yInd],p0)
        end
        if scatter
            if stdI!=false
                @df dfi scatter!(cols(xInd),cols(yInd),markersize = ms,label=lblS[i],color=namedColors[i],yerr=cols(stdI))
            else
                @df dfi scatter!(cols(xInd),cols(yInd),markersize = ms,label=lblS[i],color=namedColors[i])
            end
        else
            @df sort(dfi,:t_save) plot!(cols(xInd),cols(yInd),lw = 1,label=lblS[i])
        end
        if fitT==modelSq
            fitLbl = latexstring("fit: $(@sprintf("%.3f", fit.param[1]))","\$\\sqrt{q}\$")
        elseif fitT==modelLC
            fitLbl = latexstring("fit: $(@sprintf("%.3f", fit.param[1]))","\$q\$")
        elseif fitT==modelE
            fitLbl = latexstring("fit: $(@sprintf("%.3f", fit.param[2]))","\$q\$","^ $(@sprintf("%.3f", fit.param[1]))")
        end
        if fitT!=false 
            @df dfi plot!(cols(xInd),fitT(cols(xInd),fit.param),color=namedColors[i], label=fitLbl)
        end
    end
    plot!(xlabel=xlbl,ylabel=ylbl)
    savefig(savePath)
    return pl
end




#####
# Fitting models
#####
@. modelL(x, p) = p[1]*x+p[2]
function fitLinear(l,times)
    p0 = [1.0,0.0]
    fit = curve_fit(modelL,times,l, p0)
    return fit.param
end

@. modelE(x, p) = p[3].*x.^p[1].+p[2]
function fitExp(l,times)
    p0 = [0.5,0.0,1.0]
    fit = curve_fit(modelE,times,l, p0)
    return fit.param
end

@. modelSqrt(x, p) = p[1].*sqrt.(x).+p[2]
function fitSqrt(l,times)
    p0 = [0.0,0.0]
    fit = curve_fit(modelSqrt,times,l, p0)
    return fit.param
end
@. modelInv(x, p) = p[1]./x.+p[2]
function fitInv(l,times)
    p0 = [0.0,0.0]
    fit = curve_fit(modelInv,times,l, p0)
    return fit.param
end



"""
plot mean of v vs Ns with label lbl and save at saveLbl
plot fit as well
"""
function plotVWithFit(v,Ns,l,xlbl,ylbl,saveLbl,fit=:linear)
    function getP(f,vm,Ns,l)
        if length(l) == 1 && length(vm)>1
            p = [f(vm,Ns)]
            println(p)
        else
            p = map(vm) do vmm 
                f(vmm,Ns)
            end
        end
        return p
    end
    my_colors = ["#D43F3AFF", "#5CB85CFF", "#EEA236FF", "#46B8DAFF","#357EBDFF", "#9632B8FF", "#B8B8B8FF"]
    vm = mean(v)
    vss = my_std(v)
    if fit==:linear
        md = modelL
        p = getP(fitLinear,vm,Ns,l)
        lbl = ["fit: $(@sprintf("%.3f", p[i][1]))k+$(@sprintf("%.3f", p[i][2]))" for i=1:length(p)]
    elseif fit==:exp
        p = getP(fitExp,vm,Ns,l)
        md = modelE
        lbl = ["fit: $(@sprintf("%.3f", p[i][3]))k^$(@sprintf("%.3f", p[i][1]))+$(@sprintf("%.3f", p[i][2]))" for i=1:length(p)]
    elseif fit==:sqrt
        p = getP(fitSqrt,vm,Ns,l)
        md = modelSqrt
        lbl = ["fit: $(@sprintf("%.3f", p[i][1]))sqrt(k)+$(@sprintf("%.3f", p[i][2]))" for i=1:length(p)]
    elseif fit==:inv
        p = getP(fitInv,vm,Ns,l)
        md = modelInv
        lbl = ["fit: $(@sprintf("%.3f", p[i][1]))/k+$(@sprintf("%.3f", p[i][2]))" for i=1:length(p)]

    end

    # plot(Ns,vm,yerr=vss,seriestype=:scatter,label=l)
    # if fit!=:false
    #     plot!(Ns,md(Ns,p),lw=3,label=lbl)
    # end
    if length(l) == 1 && length(vm) > 1# if only one line to plot
        plot(Ns,vm,yerr=vss,lw=3,label=l[1],color=my_colors[1],seriestype=:scatter,legend=:outertopright,grid=false)
        if fit!=:false
            plot!(Ns,md(Ns,p[1]),lw=3,color=my_colors[1],label=lbl[1])
        end
    else
        plot(Ns,vm[1],yerr=vss[1],lw=3,label=l[1],color=my_colors[1],seriestype=:scatter,legend=:outertopright,grid=false)
        if fit!=:false
            plot!(Ns,md(Ns,p[1]),lw=3,color=my_colors[1],label=lbl[1])
        end
        if length(vm)>1
            for i=2:length(vm)
                plot!(Ns,vm[i],yerr=vss[i],lw=3,color=my_colors[i],seriestype=:scatter,label=l[i])
                if fit!=:false
                    plot!(Ns,md(Ns,p[i]),lw=3,color=my_colors[i],label=lbl[i])
                end
            end
        end
    end
    plot!(xlabel=xlbl)
    plot!(ylabel=ylbl)
    savefig(saveLbl) 
end


###############
# Calculate theoretical values 
###############

""" Calculate optimal network size for learning speed
given parameters in postStore for net with size N 
use values at interval of trainign given by interval 
"""
function getNopt(postStore,N,mu,rho,interval,s=:lms)
    # return ((2/(1+rho)).*(abs.(postStore[:gradCorr_gradO]).*postStore[:norm_gradO].*postStore[:norm_grad])./(gamma.*postStore[:hessProj_gradO].*postStore[:norm_gradO].^2)).^(1/rho)
    return N*(((1+rho)/2).*(abs.(postStore[Symbol(:gradCorr_,s)][interval]).*postStore[Symbol(:norm_,s)][interval].*postStore[:norm_grad][interval])./(mu.*postStore[Symbol(:hessProj_,s)][interval].*postStore[Symbol(:norm_,s)][interval].^2)).^(1/rho)
end

""" Calculate optimal network size for steady state error 
given parameters in postStore for net with size N 
use values at interval of trainign given by interval 
"""
function getNoptSS(postStore,N,mu,gamma,rho,interval,tr_N,s=:lms)
    # return ((2/(1+rho)).*(abs.(postStore[:gradCorr_gradO]).*postStore[:norm_gradO].*postStore[:norm_grad])./(gamma.*postStore[:hessProj_gradO].*postStore[:norm_gradO].^2)).^(1/rho)
    return N*(1/rho.*((gamma/mu)^2).*(tr_N[interval]./(postStore[Symbol(:hessProj_,s)][interval].*postStore[Symbol(:norm_,s)][interval].^2))).^(1/(rho+1))
end


""" Calculate optimal learning step 
    given parameters in postStore 
    use values at interval of training given by interval 
"""
function getmuOpt(postStore,interval,T=200,s=:lms)
    # return ((2/(1+rho)).*(abs.(postStore[:gradCorr_gradO]).*postStore[:norm_gradO].*postStore[:norm_grad])./(gamma.*postStore[:hessProj_gradO].*postStore[:norm_gradO].^2)).^(1/rho)
    return (abs.(postStore[Symbol(:gradCorr_,s)][interval]).*postStore[Symbol(:norm_,s)][interval].*postStore[:norm_grad][interval])./(postStore[Symbol(:hessProj_,s)][interval].*postStore[Symbol(:norm_,s)][interval].^2)
end


"""
    Compute the static learning speed 
    νₜ = -ΔFₜ/(δₜFₜ)
    at each training step 
"""
function computeStaticLs!(postStore,s,t_save)
    dt = hcat(vcat(1,t_save[2:end]-t_save[1:end-1])...)
    # postStore[Symbol(:staticLS_,s)] = -postStore[Symbol(:localTaskDifficulty_,s)][1,:]./(dt.*postStore[:taskError][1,:])
    postStore[Symbol(:staticLS_,s)] = -postStore[Symbol(:localTaskDifficulty_,s)]./(dt.*postStore[:taskError])
    # return postStore
end

"""
    Compute the static local task diff
    Gₜ = 1/2*γ*(||lmsₜ||/||∇Fₜ||)lmŝ∇²Fₜlmŝ
    at each training step 
"""
function computeStaticLT!(p,s,gamma)
    p[Symbol(:staticLT_,s)] = 1/2*gamma.*(p[Symbol(:norm_,s)]./p[:norm_grad]).*p[Symbol(:hessProj_,s)]
end

"""
    Compute dynamic ss error (mean over last i values of task error)
"""
function computeDynamicSS!(p,i=5,max=false,var=:taskError,name=:dynamicSS)
    if max==true
        p[name] = maximum(p[var])
    else 
        p[name] = mean(p[var][1,end-i:end])
    end
end

"""
    Compute dynamic ls mean of νₜ = -ΔFₜ/(δₜFₜ) (measure change in task loss) over the first i epochs 
"""
function computeDynamicLs!(p,t_save,i=5,var=:taskError,name=:dynamicLs)
    dt = hcat(vcat(1,t_save[2:end]-t_save[1:end-1])...)
    dF = hcat(vcat(0,p[var][1,2:end]-p[var][1,1:end-1])...)
    ls = -dF[1,:]./(dt[1,:].*p[var][1,:])
    # ls = -dF[1,:]./(dt[1,:])
    p[name] = mean(ls[3:3+i])
end

"""
    Normalise the task loss with respect to the task loss before learning 
"""
function computeNormalisedTaskE!(p,normVal=false)
    if normVal == false
        normVal = p[:taskError][1,1] # default is to normalise by initial loss
    end
    p[:taskErrorN] = p[:taskError]./normVal
end

function computeLsSSFit!(p,times, var =:taskErrorN, lsName = :lsN, ssName = :ssN)
    # fit the normalised loss get learning speed and ss 
    pN = fitLoss(p[var][1,:], times)
    p[lsName] = pN[2]
    p[ssName] = pN[3]
end

function fitLoss(loss,times,m="exp")
    if m == "exp"
        @. model(x, p) = p[1]*exp(-x*p[2])+p[3]
    end
    p0 = [loss[1],0.1,loss[end]]
    lb = [0.0, 0.0, loss[end]/1000]
    ub = [Inf, Inf, 1000*mean(loss)]
    try 
        fit = curve_fit(model,times,loss, p0, lower=lb, upper=ub)
        p = fit.param
        return p
    catch 
        println("problem with fit")
        # p = [loss[1],NaN,mean(loss[end-5:end])]
        p = [loss[1],NaN,NaN]
        return p
    end
end



""" 
    decompose vec t[s] into the direction of p[s2] plus noise
    hat{vec} = γ₁hat{ref} + γ₂hat{n}
    set in postStore p [:n_s], the noise and :gamma2_s the gamma2 
"""
function computeDecomp!(p,t,s,s2=:grad)
    gamma1, gamma2, n = decompose(t[s],p[s2])
    p[Symbol(:n_,s)] = n 
    p[Symbol(:gamma2_,s)] = gamma2
    p[Symbol(:gamma1_,s)] = gamma1
end

function computeDecomp(p,t,s,s2=:grad)
    gamma1, gamma2, n = decompose(t[s],p[s2])
    pp = Dict()
    pp[Symbol(:n_,s)] = n 
    pp[Symbol(:gamma2_,s)] = gamma2
    pp[Symbol(:gamma1_,s)] = gamma1
    return pp
end

""" 
    compute normalized dot product between vec[:,i] and ref[:,i] 
    for each i return dot product
"""
function dp(vec,ref)
    dp = zeros(1,size(vec,2))
    for i=1:size(vec,2)
        dp[:,i] .= normalize(vec[:,i])'*normalize(ref[:,i])
    end
    return dp 
end

function computeDot(p,t,s,s2=:grad)
    return dp(t[s],p[s2])
end

""" 
    compute hess proj 
"""
function hessP(vec,hess)
    hp = zeros(1,size(vec,2))
    N = length(vec[:,1])
    for i=1:size(vec,2)
        hp[:,i] .= normalize(vec[:,i])'*reshape(hess[:,i],(N,N))*normalize(vec[:,i])
    end
    return hp 
end

function computeHP(p,t,s,s2=:hessian)
    return hessP(t[s],p[s2])
end


"""
    function to generate updates from the parameters in d and weights w
"""
function makeComputeUpdate(d::Dict,W,trajErrorIndex=2)
    @unpack mus, gammas, deltaTes, deltaTrs, deltaThs, t_train, method = d
    if method == "onlineGrad+Noise"
        a = OnlineGradientCompute(mus,t_train,deltaTes,deltaTrs,W);
        b = GaussianNoiseCompute(gammas,t_train,W);
        updates =[a,b]
    elseif method == "onlineGrad"
        a = OnlineGradientCompute(mus,t_train,deltaTes,deltaTrs,W);
        updates =[a]
    elseif method == "lms"  
        updates = [LMSCompute(mus,t_trains,deltaTes,deltaTrs,deltaThs,trajErrorIndex,W)]
    end
    fulld = copy(d)
    fulld["updates"] = updates
    return fulld
end

"""
analyse a simulation with trainstore t and poststore p 
to get the correlation and hess proj
"""
function analysePost(t,p,method="onlineGrad+Noise")
    if method=="onlineGrad+Noise"||method=="onlineGrad"
        s=:gradO
    elseif method=="lms" 
        s=:lms 
    end
    # compute gamma^oe and eta^oe (proj of gradO onto grad)
    pp = computeDecomp(p,t,s,:grad) 
    # compute grad correlation dw^T\nabla F 
    pp[Symbol(:corr_,:dw)] = computeDot(p,t,:dw,:grad)
    # compute hessian projection dw^THdw, and gradO^T H gradO 
    pp[Symbol(:hessProj_,:dw)] = computeHP(p,t,:dw)
    pp[Symbol(:hessProj_,s)] = computeHP(p,t,s)
    if haskey(t,:noise)
        pp[Symbol(:hessProj_,:noise)] = computeHP(p,t,:noise)
    end
    pp[Symbol(:hessProj_,:n_,s)] = computeHP(p,pp,Symbol(:n_,s))
    return pp
end