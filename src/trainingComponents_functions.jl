###
# Helper functions for training components 
###

"""
    make callbacks and stop times for the updates, system syst and store
"""
function makeCallbacks(updates,syst,store)
    x = map(updates) do upd
        upd(syst,store)
    end
    # println(x)
    tms = map(getStopTimes, updates)
    tstopTimes = sort!(vcat(tms...))
    tstopTimes = unique(tstopTimes) 
    # println("stop times: ",tstopTimes)
    if isempty(x)
        return nothing,nothing
    else
        xUnrolled = reduce(vcat,x)
        cbs = CallbackSet(xUnrolled...)
        return cbs, tstopTimes
    end
end

"""
    return stop times during solving of ODEProblem 
    for updates OnlineGradientTrain and LMSTrain, 
    they both neeed to stop at t-te and t-tr and t for each train time t
"""
function getStopTimes(o::Union{OnlineGradientTrain,OnlineGradientCompute})
    oge = OnlineGradientExtract(o.deltaTe,o.deltaTr,o.trainTimes)
    tstopTimes = sort!(vcat(oge.te_times,oge.tr_times))
    tstopTimes = sort!(vcat(tstopTimes,o.trainTimes))
    unique(tstopTimes)
end
function getStopTimes(o::Union{LMSTrain,LMSCompute})
    oge = TrajErrorExtract(o.deltaTe,o.deltaTr,o.deltaTh,o.trainTimes,o.trajErrorIndex)
    tstopTimes = sort!(vcat(oge.te_times,oge.tr_times,oge.th_times))
    tstopTimes = sort!(vcat(tstopTimes,o.trainTimes))
    # println(tstopTimes)
    unique(tstopTimes)
end


"""
    return stop times for rest of PlasticityOperations 
    only stop at train times
"""
function getStopTimes(c::Union{PlasticityOperations,ComputePlasticityOperations})
    c.trainTimes
end    

"""
function to generate a callback for ode solver that stores variables 
    returned by extractorFs in the store with names
    dict = {name=>extractorFs} 
"""
# function makeCallback_record(times,extractorFs,names,store)
#     condition(u,t,integrator) = t ∈ times # at each train time
#     function affect!(integrator)
#         # println(integrator.t)
#         i = findfirst(x->x==integrator.t, times)
#         # println("Recording ",i)
#         map(extractorFs,names) do f,n
#             update_store!(store,f(integrator),i,n)
#         end
#     end
#     DiscreteCallback(condition,affect!)
# end

""" 
returns functions used by train callback to save values in the store 
save weights 
save train errors 
"""
function makeFOnlineRecord(sys::System,::Union{OnlineGradientTrain,LMSTrain,OnlineGradientCompute,LMSCompute,GradientTrain})
    # N = getN(sys)
    numOWeights = getNumOutputWeights(sys.nn)
    trainErrorIndex = sys.trainErrorIndex
    function getW(int)
        int.p[end-numOWeights+1:end]
    end
    function getTrainError(int)
        int.u[trainErrorIndex]
    end
    return [getTrainError, getW], [:trainError,:weights]
end

""" 
return a function that gives the weight update for online grad training
    the function takes a store and a time index
    calculates the online gradient using values saved in the store by the te and tr callbacks
    stores the online gradient in store 
    returns the update =-mu*gradO
"""
function makeFOnlineUpdate(c::Union{OnlineGradientTrain,OnlineGradientCompute})
    function updateF(i,store)
        # gradO = store[:instETr][:,i].*(store[:valTr][:,i]-store[:valTe][:,i]) # calc gradO from the store vals
        # gradO = (store[:valTr][:,i]-store[:valTe][:,i]) # calc gradO from the store vals
        gradO = (store[:valTr][:,i]-store[:valTe][:,i])./(c.deltaTe-c.deltaTr) # calc gradO from the store vals (normalize by window)
        update_store!(store,gradO,i,:gradO) # save gradO in store
        return -c.mu*store[:gradO][:,i] # weight update
    end
    return [updateF] 
end

""" 
return a function that gives the weight update for LMS training
    the function takes a store and a time index
    calculates the lms update using values saved in the store by the te and tr callbacks
    stores the update in store 
    returns the update =-mu*lms
"""
function makeFOnlineUpdate(c::Union{LMSTrain,LMSCompute}) 
    function updateF(i,store)
        N = size(store[:hTh],1)
        d = Normal(0,1/sqrt(N))
        # noise = normalize(randn(N))
        noise = zeros(N)
        # noiseScalar = randn(1)
        # etaScalar = c.eta
        noiseScalar = rand(d,1)
        etaScalar = c.eta
        # noise = rand(d,N)
        # lms = ((store[:trajETr][:,i]-store[:trajETe][:,i])/(c.deltaTe-c.deltaTr)).*store[:hTh][:,i] # calc lms from the store vals (normalize by window)
        lmsOne = ((store[:trajETr][:,i]-store[:trajETe][:,i])/(c.deltaTe-c.deltaTr) + etaScalar*noiseScalar ).*(store[:hTh][:,i].+ c.eta*noise) # calc lms from the store vals (normalize by window)
        numOutW = size(store[:lms],1) # number of output weights a multiple of N
        num_outputs = Int(numOutW/N) 
        # lms update form multiple NN outputs
        # same update for each output 
        lms = repeat(lmsOne,inner=num_outputs)
        update_store!(store,lms,i,:lms) # save lms in store
        return -c.mu*store[:lms][:,i] # weight update
    end
    return [updateF] 
end

""" 
    return a function that gives the weight update for grad training
    the function takes a store and a time index
    calculates the gradient using getGradLoss function
    stores the gradient in store 
    returns the update =-mu*grad
"""
function makeFOnlineUpdate(c::GradientTrain,system)
    prob = _buildODETestProb(system,c.trajTime) # build prob ode to later compute gradient
    if typeof(system.plant) <: NLArmPlant # nonlinear plant need differnt ode solver
        solver =  RadauIIA5()
    else
        solver = Tsit5()
    end
    pI = getInputWeights(system)
    function updateF(i,store)
        pOutput = store[:weights][:,i] # get output weights at this time
        if store[:keepTraining] # if keep training 
            taskError, grad = getGradLoss(pOutput,pI,prob,c.gradIndex,solver) # get task grad
            update_store!(store,grad,i,:grad) # save grad in store
            if taskError < c.lossCutoff # if the task loss is smaller than cutoff stop learning 
                println("stopped grad descent at $i")
                store[:keepTraining] = false
            end
            return -c.mu*grad
        else # no more training 
            return 0*pOutput # zero change in weights
        end
    end
    return [updateF] 
end

"""
function to generate a callback for ode solver that stores variables 
    returned by extractorFs in the store with names
    dict = {name=>extractorFs} 
"""
function makeCallback_record(times,extractorFs,names,store)
    # println(times)
    condition(u,t,integrator) = t ∈ times # at each train time
    # function func(u,t,integrator) # callback function 
    function affect!(integrator) # callback function 
        # println(integrator.t)
        i = findfirst(x->x==integrator.t, times)
        # println("Recording ",i)
        map(extractorFs,names) do f,n
            update_store!(store,f(integrator),i,n) # update store with output of function at index i and name n 
        end
    end
    DiscreteCallback(condition,affect!)
    # FunctionCallingCallback(func;funcat=times)
end

"""
    function to generate a callback for ode solver that updates parameters with updateF
    and stores variables returned by recordFs at store  namesR
"""
function makeCallback_update(times,updateFs,recordFs,namesR,store,N,W=false)
    condition(u,t,integrator) = t ∈ times # at each train time
    function affect!(integrator)
        # println(integrator.t)
        i = findfirst(x->x==integrator.t, times)
        # println("updating ",i)
        # store weights and train error before updating 
        map(recordFs,namesR) do f,n
            update_store!(store,f(integrator),i,n)
        end
        # get weight updates values 
        vals = map(updateFs) do f
            f(i,store)
        end
        # println("update vals", vals)
        dw = sum(vals[:,:])
        # println("dw", dw)
        update_store_sum!(store,dw,i,:dw) # update change in weights
        weightsN = length(dw) # number of weights to update
        if W==false
            integrator.p[end-weightsN+1:end] .+= dw # update weights
        else
            integrator.p[end-weightsN+1:end] = W[:,i] # update weights
        end
        # update_store_sum!(store,integrator.p[end-N+1:end],i,:weights)
    end
    DiscreteCallback(condition,affect!)
end

""" 
    returns function that takes integrator object and 
    returns local sensitivity of variable at index wrt to output weights
    to be used with callback function on a local sensititvity problem
    - numberStates: total number of states in the original ODE Problem
    - numInputWeights: number of input weights in NN (fixed parameters of the ODE)
        the gradient is taken with respect to the rest of the parameters (the output weights)
    - index: index of the variable to take the gradient with resepct to  
    - N: nuber of output wegihts to take the gradient with respect to 
"""
function makeGradExtractor(numStates,numInputWeights,index,N)
    function gradExtractor(int)
        [int.u[numStates*(numInputWeights+i)+index] for i=1:1:N]
    end
    return gradExtractor
end

""" 
    returns function that takes integrator object and 
    returns local sensitivity of variable at index wrt to output weights
    to be used with callback function on a local sensititvity problem
        take parameters from the system syst
"""
function makeGradExtractor(syst::System)
    # global onlineErrorIndex
    index = syst.onlineErrorIndex
    makeGradExtractor(syst,index)
end

""" 
    returns function that takes integrator object and 
    returns local sensitivity of variable at index wrt to output weights
    to be used with callback function on a local sensititvity problem
        take parameters from the system syst
        index: index of the variable to take the gradient with resepct to  
"""
function makeGradExtractor(syst::System,index)
    numStates = length(states(syst.system))
    numInputWeights = getNumInputWeigts(syst.nn)
    # N = getN(syst)
    numOWeights = getNumOutputWeights(syst.nn)
    makeGradExtractor(numStates,numInputWeights,index,numOWeights)
end

""" 
    function that returns function that selcts state at system.onlineErrorIndex
    from integral in callback
"""
function makeTrajErrorExtractor(syst::System,index)
    function trajErrorExtractor(int)
        int.u[index]
    end
    return trajErrorExtractor
end

function makeTrajErrorExtractor(syst::System)
    index = syst.onlineErrorIndex 
    makeTrajErrorExtractor(sys,index)
end

""" 
gets hidden layer values
"""
function makeHiddLExtractor(sys::System)
    N = getN(sys)
    mlp = sys.nn.system
    function hiddenExtractor(int)
        get_hiddens(int.sol,mlp,N)
    end
end

"""
make ODEProblem from the ode system of the system for a time from 0 to trajTime
"""
function buildOdeTrainingProblem(t::TrainingProblem,sys::System,trajTime)
    odeSyst = sys.system
    jac = eval(ModelingToolkit.generate_jacobian(odeSyst)[2])
    f = ODEFunction(odeSyst.f, jac=jac)
    # ODELocalSensitivityProblem(odeSyst,t.u0,(0.,sys.trajTime),t.pAll);
    ODEForwardSensitivityProblem(f,t.u0,(0.,trajTime),t.pAll);
end

"""
make ODEProblem from the ode system of the system for a time from 0 to trajTime
    either ODEForwardSensitivityProblem if the uptdates require adjoint gradient computation
    or ODEProblem by default
"""
function buildOdeTrainingProblem(sys::System,updates,trajTime)
    sens=false # default is doesn't require local sensitivity analysis
    for u in updates # if one of the updates requires local sens computation
        if typeof(u)<:Union{OnlineGradientTrain,OnlineGradientCompute} # if we are doing oline gradient train need local sens problem
            sens = true
        end
    end 
    odeSyst = sys.system
    u0 = zeros(length(odeSyst.states))
    if typeof(sys.plant) <: NLArmPlant
        pAll = vcat(sys.plant.param,getAllWeights(sys.nn))
    else
        pAll = getAllWeights(sys.nn)
    end
    # ODELocalSensitivityProblem(odeSyst,u0,(0.,sys.trajTime),pAll);
    if sens
        return ODEForwardSensitivityProblem(odeSyst,u0,(0.,trajTime),pAll);
    else 
        return ODEProblem(odeSyst,u0,(0.,trajTime),pAll); 
    end
end

"""
    solve the training problem with ODEProblem probSens, and callbacks
"""
function solve_prob(probSens,cbs,tstopTimes,solver=Tsit5())
    if isnothing(cbs)
        return solve(probSens, solver) 
    else
        return solve(probSens, solver, callback=cbs,tstops=tstopTimes)
        # return solve(probSens, Euler(), dt=0.001 ,callback=cbs,tstops=tstopTimes)
    end
end

"""
returns the times for gradient extraction t_train-deltaTe
"""
function getTeTimes(c::Union{OnlineGradientTrain,LMSTrain,OnlineGradientCompute,LMSCompute})
    t_obs1 = c.trainTimes .- c.deltaTe
    vcat(c.trainTimes[1],t_obs1[2:end]) # start of online error window 
end
function getTrTimes(c::Union{OnlineGradientTrain,LMSTrain,OnlineGradientCompute,LMSCompute})
    t_obs1 = c.trainTimes .- c.deltaTr
    vcat(c.trainTimes[1],t_obs1[2:end]) # start of online error window 
end

