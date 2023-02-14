###
# Different training components to update weights of NN
#   The struct <: PlasticityOperations change the weights 
#   The struct <: ComputePlasticityOperations just compute the weight change but update weights by values given
# To create a new PlasticityOperations, 
# 1. create struct for example OnlineGradientTrain
# 2. create functor (o::OnlineGradientTrain)(sys::System, store)
#    that returns an array of callback functions to pass to the ode solver
###

""" 
Online gradient trainer: train with online gradient at traintimes with learning rate mu
the online gradient is over the interval [t-deltaTe, t-deltaTr]
"""
struct OnlineGradientTrain{A,B,C,D} <: PlasticityOperations 
    mu::A 
    trainTimes::B
    deltaTe::C 
    deltaTr::D
end

""" 
Function to generate the callbacks for online grad training 
    generate the callbacks from the online grad extractor (i.e. callbacks at te_times and tr_times)
    and callback at traintimes
    return an array with the three callbacks
"""
function (o::OnlineGradientTrain)(sys::System,store)
    oge = OnlineGradientExtract(o.deltaTe,o.deltaTr,o.trainTimes) # make gradient extractor
    cbTe,cbTr = oge(sys,store) # get callback functions for times Te and Tr
    updateFs = makeFOnlineUpdate(o) # make function to update weights in callback function
    recordFs, namesR = makeFOnlineRecord(sys,o) # make function to save variable in records
    cbTrain = makeCallback_update(o.trainTimes,updateFs,recordFs,namesR,store,getNumOutputWeights(sys.nn))
    return [cbTe,cbTr,cbTrain]
end

""" 
    Online gradient compute: compute online gradient at traintimes with learning rate mu
    the online gradient is over the interval [t-deltaTe, t-deltaTr]
    parameters are updated to values given by weights (not by the online gradient update)
"""
struct OnlineGradientCompute{A,B,C,D,E} <: ComputePlasticityOperations
    mu::A 
    trainTimes::B
    deltaTe::C 
    deltaTr::D
    weights::E # array of weights to update parameters 
end

""" 
    Function to generate the callbacks for online grad compute
    generate the callbacks from the online grad extractor (i.e. callbacks at te_times and tr_times)
    and callback at traintimes
    return an array with the three callbacks
"""
function (o::OnlineGradientCompute)(sys::System,store)
    oge = OnlineGradientExtract(o.deltaTe,o.deltaTr,o.trainTimes) # make gradient extractor
    cbTe,cbTr = oge(sys,store) # get callback functions for times Te and Tr
    updateFs = makeFOnlineUpdate(o) # make function to update weights in callback function
    recordFs, namesR = makeFOnlineRecord(sys,o) # make function to save variable in records
    cbTrain = makeCallback_update(o.trainTimes,updateFs,recordFs,namesR,store,getNumOutputWeights(sys.nn),o.weights) # make callback at train times
    return [cbTe,cbTr,cbTrain]
end


"""
    Update weights in direction of 1/(deltaTe-deltaTr)*int_{t-deltaTe}^{t-deltaTr}[(y-r)]*h(t-delta t_h)    
    integral of (error)*granule cell activity over interval
"""
struct LMSTrain{A,B,C,D,E,F} <: PlasticityOperations
    mu::A
    trainTimes::B
    deltaTe::C 
    deltaTr::D
    deltaTh::E
    trajErrorIndex::F # index of int e(t) in ode called (eI)
    eta::A # noise to granule cell activity
end

"""
    Constructor no noise
"""
function LMSTrain(m,t,de,dt,dh,tei)
    LMSTrain(m,t,de,dt,dh,tei,0.0) # no noise
end

function (o::LMSTrain)(sys::System,store)
    oge = TrajErrorExtract(o.deltaTe,o.deltaTr,o.deltaTh,o.trainTimes,o.trajErrorIndex) # make extractor functions
    cbTe,cbTr,cbTh = oge(sys,store) # get callback functions for times Te and Tr
    updateFs = makeFOnlineUpdate(o) # make function to update weights in callback function
    recordFs, namesR = makeFOnlineRecord(sys,o) # make function to save variable in records
    cbTrain = makeCallback_update(o.trainTimes,updateFs,recordFs,namesR,store,getNumOutputWeights(sys.nn))
    return [cbTe,cbTr,cbTh,cbTrain] 
end

"""
    Compute weight Update in direction of 1/(deltaTe-deltaTr)*int_{t-deltaTe}^{t-deltaTr}[(y-r)]*h(t-delta t_h)  
    integral of (error)*granule cell activity over interval
    Update parameters with weights
"""
struct LMSCompute{A,B,C,D,E,F,G} <: ComputePlasticityOperations
    mu::A
    trainTimes::B
    deltaTe::C 
    deltaTr::D
    deltaTh::E
    trajErrorIndex::F
    weights::G
end
function (o::LMSCompute)(sys::System,store)
    oge = TrajErrorExtract(o.deltaTe,o.deltaTr,o.deltaTh,o.trainTimes,o.trajErrorIndex) # make extractor functions
    cbTe,cbTr,cbTh = oge(sys,store) # get callback functions for times Te and Tr
    updateFs = makeFOnlineUpdate(o) # make function to update weights in callback function
    recordFs, namesR = makeFOnlineRecord(sys,o) # make function to save variable in records
    cbTrain = makeCallback_update(o.trainTimes,updateFs,recordFs,namesR,store,getNumOutputWeights(sys),o.weights)
    return [cbTe,cbTr,cbTh,cbTrain] 
end




""" 
Add gaussian noise at trainTimes
"""
struct GaussianNoise{A,B} <: PlasticityOperations
    mu::A
    trainTimes::B 
end
function (o::GaussianNoise)(sys::System,store)
    numOWeights = getNumOutputWeights(sys)
    function updateF(i,store)
        noise = normalize(randn(numOWeights))
        # noise = randn(N)
        update_store!(store,noise,i,:noise) # save gradO in store 
        o.mu*noise
    end
    cb = makeCallback_update(o.trainTimes,[updateF],[],[],store,numOWeights)
    return [cb]
end

""" 
Add gaussian noise at trainTimes
"""
struct GaussianNoiseCompute{A,B,C} <: ComputePlasticityOperations
    mu::A
    trainTimes::B
    weights::C 
end
function (o::GaussianNoiseCompute)(sys::System,store)
    # N = getN(sys)
    numOWeights = getNumOutputWeights(sys)
    function updateF(i,store)
        noise = normalize(randn(numOWeights))
        update_store!(store,noise,i,:noise) # save gradO in store 
        o.mu*noise
    end
    cb = makeCallback_update(o.trainTimes,[updateF],[],[],store,numOWeights,o.weights)
    return [cb]
end

""" 
Gradient descent training with step mu, at trainTimes, with gradIndex
lossCutoff to stop gradient descent when loss is small enough
"""
struct GradientTrain{A,B,C,D,E} <: PlasticityOperations 
    mu::A 
    trainTimes::B
    gradIndex::C # index of parameter to take the gradient of
    trajTime::D
    lossCutoff::E
end
function GradientTrain(mu,tt,gradI,trajT)
    GradientTrain(mu,tt,gradI,trajT,0) # default zero lossCutoff
end
function (o::GradientTrain)(sys::System,store)
    updateFs = makeFOnlineUpdate(o,sys) # function to update weights
    recordFs, namesR = makeFOnlineRecord(sys,o) # make function to save variable in records
    cbTrain = makeCallback_update(o.trainTimes,updateFs,recordFs,namesR,store,getNumOutputWeights(sys)) # callback update
    return [cbTrain] 
end

 
""" 
Online training problem is a type of training problem
"""
struct OnlineTrainingProblem{A,B,C} <:TrainingProblem
    u0::A
    pAll::B 
    tstopTimes::C
    # trajTime::D 
end
function OnlineTrainingProblem(sys,updates)
    odeSyst = sys.system
    u0 = zeros(length(odeSyst.states))
    OnlineTrainingProblem(sys,updates,u0)
end
function OnlineTrainingProblem(sys::System,updates,u0)
    pAll = getAllWeights(sys.nn)
    # trajTime = sys.trajTime
    tstopTimes = getStopTimes(updates)
    OnlineTrainingProblem(u0,pAll,tstopTimes)
end

