###
# Define structures for postAnalysis: 
# after training compute quantities (task loss, task gradient etc) 
# for weights at different times during training
###

"""
Default function to execute a PostSimulationAnalysis:
    activate requirements
    run function of requirements and use returned values to compute data for PostSimulationAnalysis
    store data with name in the store
"""
function (o::PostSimulationAnalysis)(prob,t,trainStore,postStore,pI,solver)
    # find each value we need to calculate the local task otherwise calculate
    reqs, rFs = requirements(o) # required functions 
    vals = map(rFs) do f
        f(prob,t,trainStore,postStore,pI,solver) # run requirements
    end
    data = computeData(o,vals) 
    name = getName(o)
    i = findfirst( x -> x == t, o.t)  
    update_store!(postStore,data,i,name)
    return data
end


"""
compute local task difficulty=change in task error from the plasticity vector v
given by symbol in trainStore
ΔF = μv^T∇F + μ^2 v^T ∇^2F v
"""
struct LocalTaskDifficulty{A,B,C,D} <:PostSimulationAnalysis
    # calculate change in error from the change in weights in symbol with step mu
    mu::A # learning step 
    symbol::B # symbol of the variable of weight plasticity in trainStore
    t::C # times at which to save variable
    gradIndex::D
end
function requirements(c::LocalTaskDifficulty)
    req = [:norm_grad,Symbol(:norm_,c.symbol),Symbol(:gradCorr_,c.symbol),Symbol(:hessProj_,c.symbol),Symbol(:localTaskDifficulty_,c.symbol),:weights,:taskError ]
    rFs = [GradCorr(c.symbol,c.t,c.gradIndex),HessianProjection(c.symbol,c.t,c.gradIndex),GradNorm(c.t),PvecNorm(c.symbol,c.t)]
    return req,rFs
end
function computeData(o::LocalTaskDifficulty,vals)
    corr, proj, gradNorm, pVecNorm  = vals
    # println(vals)
    o.mu*corr*gradNorm*pVecNorm + o.mu^2*proj*pVecNorm^2
end
function getName(o::LocalTaskDifficulty)
    Symbol(:localTaskDifficulty_,o.symbol) 
end

"""
compute hessian and the projection of the vector to the hessian
compute traces squared etc to find the projection of the Hessian onto
 ∇L^T ∇^2F ∇L 
"""
struct HessianAnalysis{B,C,D} <:PostSimulationAnalysis
    symbol::B # symbol of the variable of weight plasticity in trainStore
    t::C # times at which to save variable
    gradIndex::D
end
function requirements(c::HessianAnalysis)
    req = [:norm_grad,Symbol(:norm_,c.symbol),Symbol(:gradCorr_,c.symbol),Symbol(:hessProj_,c.symbol),:weights,:taskError,:trace,:trace_N,:sq_tr,:cub_sq,:hessProj_grad,:hessProj_infoNoise,:infoNoise]
    rFs = [HessianC(c.symbol,c.t,c.gradIndex),PlasticityVector(c.symbol,c.t)]
    return req,rFs
end
function computeData(o::HessianAnalysis,vals)
    hessian,vec  = vals
    if norm(vec) == 0
        proj = 0
    else
        proj = normalize(vec)'*hessian*normalize(vec)
    end
    return proj
end
function getName(o::HessianAnalysis)
    Symbol(:hessProj_,o.symbol)
end

"""
dot product of normalized task error gradient 
and normalized plasticity vector given by symbol in training store
"""
struct GradCorr{A,B,C} <:PostSimulationAnalysis
    symbol::A # symbol of plasticity vector
    t::B # times at which to save var
    gradIndex::C
end
function computeData(o::GradCorr,vals)
    grad, vec = vals
    if norm(grad)==0 || norm(vec)==0
        return 0
    else
        return normalize(grad)'*normalize(vec)
    end
end
function getName(o::GradCorr)
    Symbol(:gradCorr_,o.symbol) 
end
"""
get required symbols to calculate local task LocalTaskDifficulty
"""
function requirements(o::GradCorr)
    req = [Symbol(:gradCorr_,o.symbol),:taskError]
    rFs = [Grad(o.t,o.gradIndex),PlasticityVector(o.symbol,o.t)] # calculate grad and plast vector save their norms in the stor
    return req,rFs
end


"""
    calculate learning speed and steady state error of a task error calcualted at times t
    fit with exponential function F[t] = Ae^[-a*t]+b
    the learning speed is a and the steady state error is b 
"""
struct LearningPerformanceTask{T,S} <:PostSimulationAnalysis
    t::T
    gradIndex::S
end
function (o::LearningPerformanceTask)(prob,t,trainStore,postStore,pI,solver)
    # find each value we need to calculate the local task otherwise calculate
    reqs, rFs = requirements(o)
    vals = map(rFs) do f
        f(prob,t,trainStore,postStore,pI,solver)
    end
    # at the last time step
    if t == o.t[end]
        params = fitLoss(postStore[:taskError][1,:],o.t)
        update_store!(postStore,params,:fitParams)
        update_store!(postStore,params[2],:learningSpeed)
        update_store!(postStore,params[3],:steadyStateE)
    end
end
"""
get required symbols to calculate local task LocalTaskDifficulty
"""
function requirements(o::LearningPerformanceTask)
    req = [:taskError,:fitParams,:steadyStateE,:learningSpeed]
    rFs = [TaskError(o.t,o.gradIndex)] # calculate grad and plast vector save their norms in the stor
    return req,rFs
end

"""
    calculate learning speed and steady state error of a training error (integral of traj error during training)
    fit with exponential function F[t] = Ae^[-a*t]+b
    the learning speed is a and the steady state error is b 
"""
struct LearningPerformanceTrain{A,T} <:PostSimulationAnalysis
    symbol::A # symbol of train error
    t::T # times of train error
end
function (o::LearningPerformanceTrain)(prob,t,trainStore,postStore,pI,solver)
    # at the last time step
    if t == o.t[end]
        println("getting learning performance")
        trainError = trainStore[o.symbol]
        update_store!(postStore,trainError,:trainError)
        normError = trainError'./o.t
        maxL, maxIs = findmax(normError)
        maxI = maxIs[1]
        croppedError = normError[maxI:end]
        croppedTimes = o.t[maxI:end]
        params = fitLoss(croppedError,croppedTimes)
        update_store!(postStore,params,:fitParams)
        update_store!(postStore,params[2],:learningSpeed)
        update_store!(postStore,params[3],:steadyStateE)
        update_store!(postStore,croppedError,:trainError_cropped)
        update_store!(postStore,croppedTimes,:trainTimes_cropped)
    end
end



"""
    compute the coding level as the proportion of GC that have non-zero output
    compute for weights at time t mean of coding level over all traj
        not working. can't access the symbolic variable h in the solution
        of a remake problem
"""
struct CodingLevel{T,S} <:PostSimulationAnalysis
    t::T # array of times
    NNsystem::S # ODE system of the NN related 
end
function (o::CodingLevel)(prob,t,trainStore,postStore,pI,solver)
    i = findfirst( x -> x == t, o.t)
    if i>1
        j1 = findfirst( x -> x > o.t[i-1], trainStore[:t])
    else
        j1 = 1
    end
    j = findfirst( x -> x > t, trainStore[:t]) # find index of training
    # pOutput = get_store(trainStore,:weights,j) # weights at that time 
    # tmp_prob = remake(prob,p=vcat(pI,pOutput))
    # sol = solve(tmp_prob,Tsit5())
    # cl = getCodingL(sol, o.NNsystem, length(pOutput)) # coding level of the gc layer activity
    cl = trainStore[:codingLevel][j1:j]
    codingL = [mean(cl),std(cl)]
    update_store!(postStore,codingL,i,:codingLevel) # update task error
end





"""
Get the vector for weight update at time t with symbol 
"""
struct PlasticityVector{A,B} <:PostSimulationReadout
    symbol::A 
    t::B # times at which to save var
end
function (o::PlasticityVector)(prob,t,trainStore,postStore,pI,solver)
    j = findfirst( x -> x == t, trainStore[:t_train]) # find closest train time 
    vec = get_store(trainStore,o.symbol,j) # get value of the plasticity vector at index j
    i =  findfirst( x -> x == t, o.t) 
    update_store!(postStore,norm(vec),i,Symbol(:norm_,o.symbol)) # update plasticity vec norm
    return vec
end


"""
Norm of task gradient at times t
"""
struct GradNorm{A} <:PostSimulationReadout
    t::A # times at which to save var
end
function (o::GradNorm)(prob,t,trainStore,postStore,pI,solver)
    i =  findfirst( x -> x == t, o.t) 
    get_store(postStore,:norm_grad,i)[1]
end

"""
Norm of the plasticity vector
"""
struct PvecNorm{A,B} <:PostSimulationReadout
    symbol::A
    t::B # times at which to save var
end
function (o::PvecNorm)(prob,t,trainStore,postStore,pI,solver)
    i =  findfirst( x -> x == t, o.t) 
    get_store(postStore,Symbol(:norm_,o.symbol),i)[1]
end


""" 
Trajectory if the weights were static to the value at time t
    not implemented
"""
struct PostTrajectories{A} <: PostSimulationAnalysis
    t::A
end
function (o::PostTrajectories)(prob,t,trainStore,postStore,pI,solver)
    j = findfirst( x -> x == t, trainStore[:t_train])
    # println("Computing grad ", j)
    pOutput = get_store(trainStore,:weights,j) 
    tmp_prob = remake(prob,p=vcat(pI,pOutput))
    # sol = solve(tmp_prob,solver)
    i = findfirst( x -> x == t, o.t) 
    update_store!(postStore,pOutput,i,:weights) # update task error 
    # update_store!(postStore,sol[],i,:trajectory)
end

"""
Compute gradient of task loss (task loss is given at index gradIndex)
"""
struct Grad{T,S} <: PostSimulationReadout
    t::T 
    gradIndex::S
end
""" 
    get gradient of problem prob 
    with weights from trainstore :weights at at index j given by time t in :t_train
    update postsotere with task error, norm of gradient, weights and return gradient
"""
function (o::Grad)(prob,t,trainStore,postStore,pI,solver)
    j = findfirst( x -> x == t, trainStore[:t_train])
    # println("Computing grad ", j)
    pOutput = get_store(trainStore,:weights,j)
    # println(pOutput)
    taskError, grad = getGradLoss(pOutput,pI,prob,o.gradIndex,solver)
    i = findfirst( x -> x == t, o.t) 
    update_store!(postStore,pOutput,i,:weights) # update task error
    update_store!(postStore,taskError,i,:taskError) # update task error
    update_store!(postStore,norm(grad),i,:norm_grad) # update grad norm
    update_store!(postStore,grad,i,:grad) # update grad norm
    return grad
    # update_store!(postStore,taskError,i,:taskError)
end

"""
Compute task loss 
"""
struct TaskError{T,S} <: PostSimulationReadout
    t::T
    gradIndex::S
end
function (o::TaskError)(prob,t,trainStore,postStore,pI,solver)
    j = findfirst( x -> x >= t, trainStore[:t_train])
    # println(j)
    pOutput = get_store(trainStore,:weights,j)
    taskError = getTaskError(pOutput,pI,prob,o.gradIndex,solver)
    i = findfirst( x -> x == t, o.t) 
    update_store!(postStore,taskError,i,:taskError) # update task error
    return taskError
end

"""
Compute projecton of the vector saved in trainstore with symbol adn the hessian of task loss
"""
struct HessianProjection{A,T,S} <: PostSimulationReadout
    symbol::A
    t::T
    gradIndex::S
end
function (o::HessianProjection)(prob,t,trainStore,postStore,pI,solver)
    j = findfirst( x -> x == t, trainStore[:t_train])
    # println("computing hessian ", j)
    # println(j)
    pOutput = get_store(trainStore,:weights,j)
    val = get_store(trainStore,o.symbol,j)
    if norm(val)==0
        proj = 0
    else
        proj = gradHessProj(pOutput,val,prob,pI,o.gradIndex,solver)
    end 
    i = findfirst( x -> x == t, o.t)
    # println(size(proj))
    update_store!(postStore,proj,i,Symbol(:hessProj_,o.symbol)) # update hessian projection
    return proj 
end

"""
Full hessian analysis. 
Compute the hessian of the task loss 
and task loss, gradient, and hessian projections
"""
struct HessianC{A,T,S} <: PostSimulationReadout
    symbol::A
    t::T
    gradIndex::S
end
function (o::HessianC)(prob,t,trainStore,postStore,pI,solver)
    j = findfirst( x -> x == t, trainStore[:t_train])
    # println("computing hessian ", j)
    # println(j)
    pOutput = get_store(trainStore,:weights,j)
    val = get_store(trainStore,o.symbol,j) # plasticity vector
    taskError,grad,hessian = getGradAndHess(pOutput,pI,prob,o.gradIndex,solver)
    i = findfirst( x -> x == t, o.t)
    update_store!(postStore,taskError,i,:taskError) # update taskError 
    update_store!(postStore,pOutput,i,:weights) # update weights 
    update_store!(postStore,norm(grad),i,:norm_grad) # update grad norm
    update_store!(postStore,grad,i,:grad) # update grad norm
    update_store!(postStore,normalize(grad)'*hessian*normalize(grad),i,:hessProj_grad) # update grad norm
    if norm(grad)==0 || norm(val)==0
        infoNoise = zeros(size(grad))
        hpInfoNoise = 0
        corr = 0
    else
        corr = normalize(grad)'*normalize(val)
        infoNoise = normalize(val) - corr.*normalize(grad)
        hpInfoNoise = normalize(infoNoise)'*hessian*normalize(infoNoise)
    end
    update_store!(postStore,corr,i,Symbol(:gradCorr_,o.symbol)) # update grad norm
    update_store!(postStore,hpInfoNoise,i,:hessProj_infoNoise) # update grad norm
    update_store!(postStore,infoNoise,i,:infoNoise) # update grad norm
    update_store!(postStore,[hessian...],i,:hessian) # update grad norm
    hessianAnalysis(o,postStore,hessian,i) # save hessian terms we are interested in 
    return hessian
end
"""
Compute the terms of hessian eigenvalues (trace, distribution... )
"""
function hessianAnalysis(o::HessianC,postStore,hessian,i)
    trace = tr(hessian)
    eigenvalues = eigvals(hessian)
    function cub(x)
        x^3
    end
    eigSq = sum(abs2,eigenvalues)
    eigCub = sum(cub,eigenvalues)
    trace_N = trace/size(hessian,1)
    sq_Tr =  eigSq/trace
    cub_sq = eigCub/eigSq
    update_store!(postStore,trace,i,:trace) # update taskError 
    update_store!(postStore,trace_N,i,:trace_N) # update taskError 
    update_store!(postStore,sq_Tr,i,:sq_tr) # update weights 
    update_store!(postStore,cub_sq,i,:cub_sq) # update grad norm
end
