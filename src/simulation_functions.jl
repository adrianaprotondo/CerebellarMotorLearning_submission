###
# Functions for simulating the motor control system 
###

"""
    Train system defined by system with updates 
    build the ode sensitivy problem based on system 
    build store to store variables while training 
    return the store with the variables during training 
"""
function train(system,updates,trajTime=false;computeCL=true)
    if trajTime==false # use trajectory time of the system
        probSens = buildOdeTrainingProblem(system,updates,system.trajTime)
    else 
        probSens = buildOdeTrainingProblem(system,updates,trajTime)
    end
    # build store to save training variables
    store = build_store(system,updates)
    # callback functions from updates and times at which to stop
    cbs, tstopTimes = makeCallbacks(updates,system,store)
    
    if typeof(system.plant) <: NLArmPlant # nonlinear plant need differnt ode solver
        solver =  RadauIIA5()
    else
        solver = Tsit5()
    end

    solOnline = solve_prob(probSens,cbs,tstopTimes,solver) # solve ODE

    println("Plotting")

    ### Add plant output to the store ####
    recordDataIndex = [system.outputIndex] # plant output 
    # recordDataIndex = [system.plant.system.o[1]] # plant output 
    recordDataN = [:plantOutput]
    # add plant output to the store
    extract_solData!(solOnline,recordDataIndex,recordDataN,store)

    ### Add reference traj to the store ###
    refTraj = system.ref.func.(solOnline.t)
    update_store!(store,refTraj,:refTraj) # add reference trajectory to the store

    if computeCL
        ### Add coding level to the store ###
        codingL = getCodingL(solOnline,system,getN(system))
        update_store!(store,codingL,:codingLevel) # add coding level
    end
    return store
end

"""
    Run post training analysis on system with training values in trainstore
    apply post-process in records array at times t_save 
    return store with post process variables
    trajTime if the traj time is different to the system traj time
        important to compute the loss and gradients over the whole new traj time
"""
function postProcess(system,trainStore,t_save,records,trajTime=false)
    postStore = build_store(system,records) # build store
    probF = buildODETestProbF(system,trajTime) 
    j = findfirst( x -> x == t_save[1], trainStore[:t_train])
    W = get_store(trainStore,:weights,j) # get initial weights 
    prob = probF(W) # build problem with weights of first no need to rebuild problem as we give weights at each time as parameter
    if typeof(system.plant) <: NLArmPlant # nonlinear plant need differnt ode solver
        solver =  RadauIIA5()
    else
        solver = Tsit5()
    end
    map(t_save) do t 
        map(records) do el # for each post-process 
            el(prob,t,trainStore,postStore,getInputWeights(system),solver)
            # at each time point of interest
        end
    end
    postStore[:t_save] = t_save
    return postStore
end

"""
Build ode problem for post-process setting the output weights of the nn to W 
"""
function buildODETestProb(sys,W,trajTime=false)
    odeSyst = sys.system
    u0 = zeros(length(odeSyst.states))
    pAll = getAllWeights(sys.nn)
    N = getN(sys)
    pAll[end-N+1:end] = W # set output weights to new output weights
    if trajTime==false # use trajTime of the system 
        return ODEProblem(sys.system, u0, (0.,sys.trajTime),pAll)
    else 
        return ODEProblem(sys.system, u0, (0.,trajTime),pAll) 
    end
end

"""
Build ODE problem for post-process
"""
function _buildODETestProb(sys,trajTime=false)
    odeSyst = sys.system
    u0 = zeros(length(odeSyst.states))
    if typeof(sys.plant) <: NLArmPlant
        pAll = vcat(sys.plant.param,getAllWeights(sys.nn))
    else
        pAll = getAllWeights(sys.nn)
    end
    if trajTime==false
        return ODEProblem(sys.system, u0, (0.,sys.trajTime),pAll)
    else 
        return ODEProblem(sys.system, u0, (0.,trajTime),pAll) 
    end
end

"""
Return function that generates ODEProblem for post-process for weights w
"""
function buildODETestProbF(sys,trajTime=false)
    odeSyst = sys.system
    u0 = zeros(length(odeSyst.states))
    pAll = getAllWeights(sys.nn)
    N = getN(sys)
    pI = pAll[1:end-N]  # input weights
    if typeof(sys.plant) <: NLArmPlant
        pStart = vcat(sys.plant.param,pI)
    else
        pStart = pI 
    end
    function probF(W)
        if trajTime==false
            return ODEProblem(odeSyst, u0, (0.,sys.trajTime),vcat(pStart,W))
        else 
            return ODEProblem(odeSyst, u0, (0.,trajTime),vcat(pStart,W))
        end
    end
    return probF
end

###############
# Simulate
###############
"""
simulate system with updates and records saving at t_save 
trajTimePP if we want a different traj time for the post process to the default system trajTime
"""
function simulate(system,updates,t_save,records,trajTime=false,trajTimePP=false,savePath="../Figures/deltaTe/";computeCL=true)
    N = getN(system) # number of output weights
    println("N=",N)
    # deltaTe = updates[1].deltaTe
    if ~isempty(updates)
        gamma = updates[1].mu
        println("gamma=", gamma) #print learning step
        println("training")
        saveLbl = "N-$N _gamma-$gamma"
    else 
        saveLbl = "N-$N"
    end 

    if trajTime==false # use default trajectory time
        trainStore = train(system,updates,system.trajTime;computeCL=computeCL) # train
    else 
        trainStore = train(system,updates,trajTime;computeCL=computeCL) # train
    end

    # plot reference traj and actual traj
    plot(trainStore[:t],trainStore[:refTraj],lw=3,label="ref")
    plot!(trainStore[:t],trainStore[:plantOutput],lw=3,label="actual traj")
    plot!(xlabel="trajectory time")
    savefig(string(savePath,"trainTraj_",saveLbl,".pdf"))

    if computeCL
        # plot coding level
        plot(trainStore[:t],trainStore[:codingLevel],lw=3)
        plot!(xlabel="trajectory time", ylabel="coding level")
        savefig(string(savePath,"codingLevel_",saveLbl,".pdf"))
    end

    if ~isempty(updates)
        # plot weights
        plot(trainStore[:weights][1:N,:]',lw=3)
        plot!(xlabel="trajectory time",ylabel="weights")
        savefig(string(savePath,"weights_",saveLbl,".pdf"))

        # plot change in weights
        plot([norm(trainStore[:dw][:,i]) for i=1:size(trainStore[:dw],2)],lw=3)
        plot!(xlabel="trajectory time",ylabel="weight update norm")
        savefig(string(savePath,"dw_",saveLbl,".pdf"))
    end

    # do post-process 
    println("testing")
    postStore = postProcess(system,trainStore,t_save,records,trajTimePP)
    map(records) do r
        plotSummary(r,trainStore,postStore,t_save,N,saveLbl,savePath)
    end
    return trainStore, postStore
end

###############
# Plot while simulating
###############
"""
    plots to save if we did learningPerformanceTask called by simulate function
        plot the task loss and the fit
"""
function plotSummary(o::LearningPerformanceTask,trainStore,postStore,t_save,N,saveLbl,savePath="../Figures/deltaTe/")
    @. model(x, p) = p[1]*exp(-x*p[2])+p[3] # task loss fit model 
    plot(t_save,postStore[:taskError]',lw=3,label=string("task error N=",N))
    params = postStore[:fitParams]
    plot!(t_save,model(t_save,params),lw=3,ls=:dash,label=string("fit task error N=",N))
    plot!(xlabel="trajectory time",ylabel="task loss")
    savefig(string(savePath,"taskError_",saveLbl,".pdf")) 
end

"""
    plots to save if we did learningPerformanceTrain called by simulate function
        plot training error 
"""
function plotSummary(o::LearningPerformanceTrain,trainStore,postStore,t_save,N,saveLbl,savePath="../Figures/deltaTe/")
    plot(trainStore[:t_train],trainStore[:trainError]'./trainStore[:t_train],lw=3,label=string("train error N=",N))
    plot!(xlabel="trajectory time")
    savefig(string(savePath,"trainError_",saveLbl,".pdf")) 
end

"""
    plots to save if we did localTaskDifficulty called by simulate function
        plot task loss
        plot gradient correlation of dw
        plot hessian projection of dw 
        plot local task difficulty
"""
function plotSummary(o::LocalTaskDifficulty,trainStore,postStore,t_save,N,saveLbl,savePath="../Figures/deltaTe/")
    plot(t_save,postStore[:taskError]',lw=3,label=string("task error N=",N))
    plot!(xlabel="trajectory time")
    savefig(string(savePath,"taskError_",saveLbl,".pdf")) 

    t_plot = t_save[2:end]
    plot(t_plot,postStore[Symbol(:gradCorr_,o.symbol)][1,2:end],lw=3,label=string("gradient correlation N = ",N))
    plot!(xlabel="trajectory time")
    savefig(string(savePath,"gradCorr_",saveLbl,".pdf"))

    t_plot = t_save[2:end]
    plot(t_plot,abs.(postStore[Symbol(:gradCorr_,o.symbol)])[1,2:end],lw=3,label=string("abs gradient correlation N = ",N))
    plot!(xlabel="trajectory time")
    savefig(string(savePath,"gradCorrAbs_",saveLbl,".pdf"))

    plot(t_plot,postStore[Symbol(:hessProj_,o.symbol)][1,2:end],lw=3,label=string("hessian projection N=",N))
    plot!(xlabel="trajectory time")
    savefig(string(savePath,"hessProj_",saveLbl,".pdf"))

    plotInt = t_save[2:end]-t_save[1:end-1]
    plot(t_plot,postStore[Symbol(:localTaskDifficulty_,o.symbol)][1,2:end]./plotInt,lw=3,label=string("change in error N=",N))
    plot!(xlabel="trajectory time")
    savefig(string(savePath,"changeError_",saveLbl,".pdf"))
end

"""
    plots to save if we did hessianAnalysis called by simulate function
        plot hessian projection of dw, gradient, online learning error, and trace terms
        plot hessian projection of dw and trace terms
"""
function plotSummary(o::HessianAnalysis,trainStore,postStoreH,t_save,N,saveLbl,savePath="../Figures/deltaTe/")
    plot(t_save[2:end],postStoreH[Symbol(:hessProj_,o.symbol)][2:end],lw=3,label=L"$\hat{\nabla L}^T\nabla^2F\hat{\nabla L}$")
    plot!(t_save[2:end],postStoreH[:hessProj_grad][2:end],lw=3,label=L"$\hat{\nabla F}^T\nabla^2F\hat{\nabla F}$")
    plot!(t_save[2:end],postStoreH[:hessProj_infoNoise][2:end],lw=3,label=L"$\hat{n}^{oe}\nabla^2F\hat{n}^{oe}$")
    plot!(t_save,postStoreH[:trace_N]',lw=3,label=L"\frac{Tr(H)}{N}")
    # plot!(t_save,postStoreH[:sq_tr]',lw=3,label="Tr2/Tr")
    plot!(t_save,postStoreH[:cub_sq]',lw=3,label=L"\frac{Tr(H^3)}{Tr(H^2)}")
    plot!(xlabel="trajectory time")
    savefig(string(savePath,"hessianAnalysis_all_",saveLbl,".pdf"))
    
    plot(t_save[2:end],postStoreH[Symbol(:hessProj_,o.symbol)][2:end],lw=3,label=L"$\hat{\nabla L}^T\nabla^2F\hat{\nabla L}$")
    # plot!(t_saveH,postStoreH[:trace_N]',lw=3,label="trace/N")
    plot!(t_save,postStoreH[:sq_tr]',lw=3,label=L"\frac{Tr(H^2)}{Tr(H)}")
    plot!(t_save,postStoreH[:cub_sq]',lw=3,label=L"\frac{Tr(H^3)}{Tr(H^2)}")
    plot!(xlabel="trajectory time")
    savefig(string(savePath,"hessianAnalysis_gradO_",saveLbl,".pdf"))

    plot(t_save[2:end],postStoreH[:hessProj_grad][2:end],lw=3,label=L"$\hat{\nabla F}^T\nabla^2F\hat{\nabla F}$")
    # plot!(t_saveH,postStoreH[:trace_N]',lw=3,label="trace/N")
    plot!(t_save,postStoreH[:sq_tr]',lw=3,label=L"\frac{Tr(H^2)}{Tr(H)}")
    plot!(t_save,postStoreH[:cub_sq]',lw=3,label=L"\frac{Tr(H^3)}{Tr(H^2)}")
    plot!(xlabel="trajectory time")
    savefig(string(savePath,"hessianAnalysis_grad_",saveLbl,".pdf"))
end
"""
    plots to save if we did learningPerformanceTask called by simulate function
        plot the task loss and the fit
"""
function plotSummary(o::CodingLevel,trainStore,postStore,t_save,N,saveLbl,savePath="../Figures/deltaTe/")
    plot(t_save,postStore[:codingLevel][1,:],yerr=postStore[:codingLevel][2,:],lw=3,label=string("coding level N=",N))
    plot!(xlabel="trajectory time",ylabel="coding level")
    savefig(string(savePath,"codingLevelPost_",saveLbl,".pdf")) 
end