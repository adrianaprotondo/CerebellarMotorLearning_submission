###
# functions for making the store (dictionary) to save variable while training and postAnalysis
###

###
# Training store
###
""" 
Initialise store for updates, and records
    initialise to empty arrays the elements for each update
"""
function build_store(sys,updates)
    store = Dict{Symbol,Any}()
    for el in updates
        add_to_store!(el,store,getNumOutputWeights(sys.nn),getN(sys))
    end
    return store 
end

function build_store(sys::InputSystem)
    store = Dict{Symbol,Any}()
    return store 
end

"""
Default variables to save for PlasticityOperations
"""
function _add_to_store!(c::Union{PlasticityOperations,ComputePlasticityOperations},store,N)
    get!(store,:dw,zeros(N,length(c.trainTimes))) # weight update computed by learning rule
    get!(store,:weights,zeros(N,length(c.trainTimes))) # weights trajectory of NN 
    get!(store,:trainError,zeros(1,length(c.trainTimes))) # training error (integral of traj error)
    get!(store,:t_train,c.trainTimes) # train times
end
"""
    For online gradient descent 
"""
function add_to_store!(c::Union{OnlineGradientTrain,OnlineGradientCompute},store,N,Nh)
    _add_to_store!(c,store,N) 
    get!(store,:valTe,zeros(N,length(getTeTimes(c))))
    get!(store,:valTr,zeros(N,length(getTrTimes(c))))
    get!(store,:instETr, zeros(1,length(getTrTimes(c))))
    get!(store,:gradInstTr,zeros(N,length(getTrTimes(c))))
    get!(store,:gradO,zeros(N,length(c.trainTimes))) # online gradient computed
end
"""
    For gaussian noise 
"""
function add_to_store!(c::Union{GaussianNoise,GaussianNoiseCompute},store,N,Nh)
    _add_to_store!(c,store,N)
    get!(store,:noise,zeros(N,length(c.trainTimes)))
end
"""
    For LMS training 
"""
function add_to_store!(c::Union{LMSTrain,LMSCompute},store,N,Nh)
    _add_to_store!(c,store,N) 
    get!(store,:trajETe,zeros(1,length(getTeTimes(c))))
    get!(store,:trajETr,zeros(1,length(getTrTimes(c))))
    get!(store,:hTh,zeros(Nh,length(getTrTimes(c)))) # hidden layer activity (array is too big if more than one NN output)
    # get!(store,:gradInstTr,zeros(N,length(getTrTimes(c))))
    get!(store,:lms,zeros(N,length(c.trainTimes))) # lms weight update 
end
"""
    For gradient descent
"""
function add_to_store!(c::GradientTrain,store,N,Nh)
    _add_to_store!(c,store,N) # base case variables 
    get!(store,:keepTraining,true) # variable to stop training after cutoff task loss
    get!(store,:grad,zeros(N,length(c.trainTimes)))
end

###
# PostAnalysis store
###
function _add_to_store!(c::Union{LocalTaskDifficulty,HessianAnalysis,GradCorr},store,N)
    get!(store,:norm_grad,zeros(1,length(c.t))) # norm of task gradient
    get!(store, Symbol(:norm_,c.symbol) ,zeros(1,length(c.t))) # norm of change in weights vector
    get!(store,:weights, zeros(N,length(c.t)))  # weights 
    get!(store,:taskError, zeros(1,length(c.t))) # task loss
    get!(store, :grad ,zeros(N,length(c.t)))  # gradient
end
function add_to_store!(c::LocalTaskDifficulty,store,N,Nh)
    _add_to_store!(c,store,N)
    get!(store,Symbol(:gradCorr_,c.symbol),zeros(1,length(c.t))) # correlation of gradient and vector
    get!(store,Symbol(:hessProj_,c.symbol),zeros(1,length(c.t))) # hessian projection
    get!(store, Symbol(:localTaskDifficulty_,c.symbol) ,zeros(1,length(c.t)))  # local task difficulty 
end
function add_to_store!(c::HessianAnalysis,store,N,Nh)
    _add_to_store!(c,store,N)
    get!(store,Symbol(:hessProj_,c.symbol),zeros(1,length(c.t))) # hessian projection
    get!(store,Symbol(:gradCorr_,c.symbol),zeros(1,length(c.t))) # correlation of gradient and vector
    get!(store,:hessProj_grad,zeros(1,length(c.t))) # hessian projection
    get!(store,:hessProj_infoNoise,zeros(1,length(c.t))) # hessian projection
    get!(store,:infoNoise,zeros(N,length(c.t))) # information error
    get!(store,:trace, zeros(1,length(c.t)))
    get!(store,:trace_N, zeros(1,length(c.t)))
    get!(store,:sq_tr, zeros(1,length(c.t)))
    get!(store,:cub_sq, zeros(1,length(c.t)))
    get!(store,:hessian, zeros(N*N,length(c.t)))
end
function _add_to_store!(::Union{LearningPerformanceTrain,LearningPerformanceTask},store,N)
    get!(store,:fitParams,nothing)
    get!(store,:steadyStateE,nothing)
    get!(store,:learningSpeed,nothing)
end
function add_to_store!(c::LearningPerformanceTrain,store,N,Nh)
    _add_to_store!(c,store,N)
    get!(store,:trainError,nothing) 
end
function add_to_store!(c::LearningPerformanceTask,store,N,Nh)
    _add_to_store!(c,store,N)
    get!(store,:taskError,zeros(1,length(c.t)))
end
function add_to_store!(c::GradCorr,store,N,Nh)
    _add_to_store!(c,store,N)
    get!(store,Symbol(:gradCorr_,c.symbol),zeros(1,length(c.t))) # correlation of gradient and vector
end
function add_to_store!(c::CodingLevel,store,N,Nh)
    get!(store,:codingLevel,zeros(2,length(c.t))) # add coding level (mean and std over whole trajectory)
end




function update_store!(store,val,name)
    store[name] = val
end
function update_store!(store,val,i,name)
    # println("updating ",name)
    # println(val)
    store[name][:,i] .= val
end
function update_store_sum!(store,val,i,name)
    store[name][:,i] .+= val
end
function get_store(store,name,i)
    store[name][:,i]
end


"""
    extract the recordData from the ODE solution solOnline and save in store
"""
function extract_solData!(solOnline,recordDataIndex,recordDataN,store)
    update_store!(store,solOnline.t,:t) 
    map(recordDataIndex,recordDataN) do i,name
        update_store!(store,solOnline[i,:],name)
    end 
end