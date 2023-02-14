###
# Defines components for the motor control system 
###
abstract type InSimulationOperations end
abstract type PlasticityOperations <: InSimulationOperations end
abstract type ComputePlasticityOperations <: InSimulationOperations end
abstract type PostSimulationAnalysis end
abstract type PostSimulationReadout end
abstract type Plant end
abstract type TrainingProblem end
abstract type FBController end

struct RandomInitialization end

""" 
System: trajectory tracking system with plant, fb controller, cerebellar-like net and reference
"""
struct System{A <: Plant,B,C,D,E,F,T,G}
    plant::A 
    pid::B
    nn::C
    ref::D
    trajTime::E
    lookahead_times::F
    system::T
    trainErrorIndex::G # integral of mse (compute task loss wrt to this)
    onlineErrorIndex::G # integral of e(t) = y(t)-r(t) for online training
    outputIndex::G # index of output in the system ode
end

""" 
Consturctor of the system give plant, 
    pidgains, neural network, reference function 
    output_index depends on the number of independent states of the plant
"""
function System(plant::Plant,pidGains,nn,refF,trajTime,lookahead_times,output_index)
    pid = PID(pidGains) # create PID controller from the given gains 
    ref = Reference(refF,lookahead_times) # reference system from the function refF
    N = nn.dims[2] # number of granule cells (i.e. adaptable parameters in the nn)
    all = connectFFFB(plant.system,ref.system,pid.system,nn.system,lookahead_times,N) # connect different systems for ODE
    system = structural_simplify(all) # simplify ODE system 
    System(plant,pid,nn,ref,trajTime,lookahead_times,system,1,1,output_index) # initialize System with default indeces
end

""" 
Consturctor of the system given plant matrices for linear system, 
    pidgains, neural network, reference function 
"""
function System(plantMatrices,pidGains,nn,refF,trajTime,lookahead_times)
    plant = LTIPlant(plantMatrices) # create plant
    System(plant,pidGains,nn,refF,trajTime,lookahead_times,3) # default index for output is 3 
end

""" 
Consturctor of the system given plant matrices for linear system, 
    pidgains, dimensions of the neural net (no nn given)
"""
function System(plantMatrices,pidGains,nnDims,K,refF,trajTime,lookahead_times;actF=tanh,varB=0.0)
    nn = NeuralNetwork(RandomInitialization(),nnDims,K;actF=actF,varB=varB)   # initialize NN with random weights
    System(plantMatrices,pidGains,nn,refF,trajTime,lookahead_times)
end


""" 
Consturctor of the system given plant, 
    pidgains, dimensions of the neural net (no nn given)
    output_index depends on the number of independent states of the plant
"""
function System(plant::Plant,pidGains,nnDims,K,refF,trajTime,lookahead_times,output_index;actF=tanh,varB=0.0)
    nn = NeuralNetwork(RandomInitialization(),nnDims,K;actF=actF,varB=varB)   # initialize NN with random weights
    System(plant,pidGains,nn,refF,trajTime,lookahead_times,output_index)
end


""" 
Consturctor of the system given plant matrices, pidgains, dimensions of the neural net
and the input Z and output weight W matrices of the neural net
"""
function System(plantMatrices,pidGains,nnDims,refF,trajTime,lookahead_times,Z,W;actF=tanh,varB=0.0)
    nn = NeuralNetwork(nnDims,Z,W;actF=actF,varB=varB)  # initialize NN with given weights
    System(plantMatrices,pidGains,nn,refF,trajTime,lookahead_times)
end

""" 
Consturctor of the system given plant, pidgains, dimensions of the neural net
and the input Z and output weight W matrices of the neural net
"""
function System(plant::Plant,pidGains,nnDims,refF,trajTime,lookahead_times,Z,W,output_index;actF=tanh,varB=0.0)
    nn = NeuralNetwork(nnDims,Z,W;actF=actF,varB=varB)  # initialize NN with given weights
    System(plant,pidGains,nn,refF,trajTime,lookahead_times,output_index)
end


""" 
function to instantiate a system 
"""
function build_system(plantMatrices,Ks,nnDims,K,refF,trajTime,lookahead_times;actF=tanh,varB=0.0)
    System(plantMatrices,Ks,nnDims,K,refF,trajTime,lookahead_times;actF=actF,varB=varB)
end

""" 
function to instantiate a system with the weights matrices for NN
"""
function build_system(plantMatrices,pidGains,nnDims,refF,trajTime,lookahead_times,Z,W;actF=tanh,varB=0.0)
    System(plantMatrices,pidGains,nnDims,refF,trajTime,lookahead_times,Z,W;actF=actF,varB=varB)
end

function build_system(plant::Plant,pidGains,nnDims,K,refF,trajTime,lookahead_times,output_index;actF=tanh,varB=0.0)
    System(plant,pidGains,nnDims,K,refF,trajTime,lookahead_times,output_index;actF=actF,varB=varB)
end

function build_system(plant::Plant,pidGains,nnDims,refF,trajTime,lookahead_times,Z,W,output_index;actF=tanh,varB=0.0)
    System(plant,pidGains,nnDims,refF,trajTime,lookahead_times,Z,W,output_index;actF=actF,varB=varB)
end

function build_system(plant::Plant,pid::FBController,nnDims,ref,trajTime,lookahead_times,Z,W,output_index;actF=tanh,varB=0.0)
    nn = NeuralNetwork(nnDims,Z,W;actF=actF,varB=varB)
    N = nn.dims[2] # number of granule cells (i.e. adaptable parameters in the nn)
    all = connectFFFB(plant.system,ref.system,pid.system,nn.system,lookahead_times,N) # connect different systems for ODE
    syst = structural_simplify(all) # simplify ODE system 
    System(plant,pid,nn,ref,trajTime,lookahead_times,syst,1,1,output_index)
end



""" 
InputSystem: system with only reference generator and neural net
"""
struct InputSystem{C ,D ,E ,F}
    nn::C ## flux net with two layers
    ref::D ## function generating input at time t
    trajTime::E
    lookahead_times::F
end
# """ 
# Consturctor of the system given neural network... 
# """
# function InputSystem(nn,fc::Float64,trajTime,lookahead_times)
#     refSin = sinFunDef(fc) # superposition of sins with cutoff frec fc
#     # ref = Reference(refSin,lookahead_times)
#     InputSystem(nn,refSin,trajTime,lookahead_times)
# end
""" 
Consturctor of the system given dimensions of the neural net
"""
function InputSystem(nnDims,K,refF,trajTime,lookahead_times;actF=actF,varB=0)
    println(K)
    Z = createZ(nnDims[1],nnDims[2],K)
    d = Normal(0,1/sqrt(nnDims[2]))
    W = rand(d,(nnDims[3],nnDims[2]))
    InputSystem(nnDims,refF,trajTime,lookahead_times,Z,W;actF=actF,varB=varB)
end
""" 
Consturctor of the system given dimensions of the neural net
and the input Z and output weight W matrices of the neural net
actF is the activation function for the NN 
varB is the variance of the bias
"""
function InputSystem(nnDims,refF,trajTime,lookahead_times,Z,W;actF=tanh,varB=0)
    # println(Z)
    nn = Chain(
        Dense(Z, true, actF),
        Dense(W,false,identity)
    )
    # set the biases of the NN 
    if varB>0.0
        # nn[1].bias .= rand(Normal(0,varB),nnDims[2])
        # nn[1].bias .= abs.(rand(Normal(0,varB),nnDims[2]))
        # nn[1].bias .= varB*ones(nnDims[2])
        nn[1].bias .= rand(Uniform(0.0,varB),nnDims[2])
    else 
        nn[1].bias .= rand(Uniform(varB,0.0),nnDims[2])
    end
    InputSystem(nn,refF,trajTime,lookahead_times)
end
""" 
function to instantiate an input system 
"""
function build_inputSystem(nnDims,K,refF,trajTime,lookahead_times;actF=tanh,varB=0)
    # println("build 1")
    InputSystem(nnDims,K,refF,trajTime,lookahead_times;actF=actF,varB=varB)
end
""" 
function to instantiate an input system 
"""
function build_inputSystem(nnDims,refF,trajTime,lookahead_times,Z,W;actF=tanh,varB=0)
    InputSystem(nnDims,refF,trajTime,lookahead_times,Z,W;actF=actF,varB=varB)
end
# """ 
# function to instantiate a system given some plant, pid and ref already instantiated
# """
# function build_inputSystem(nnDims,ref,trajTime,lookahead_times,Z,W)
#     nn = Chain(
#         Dense(Z, true, tanh),
#         Dense(W,false,identity)
#     )
#     InputSystem(nn,ref,trajTime,lookahead_times)
# end

""" 
NeuralNetwork: cerebellar like network with one hidden layer
"""
struct NeuralNetwork{A,B,C,D,T}
    dims::A
    Z::B
    W::C
    fluxNet::D
    system::T
end
""" 
Neural net constructor for random initialisation with dim and K input degree of each hidden unit
"""
function NeuralNetwork(::RandomInitialization,nnDims,K;actF=tanh,varB=0.0)
    Z = createZ(nnDims[1],nnDims[2],K)
    d = Normal(0,1/sqrt(nnDims[2]))
    W = rand(d,(nnDims[3],nnDims[2]))
    NeuralNetwork(nnDims,Z,W;actF=actF,varB=varB)
end
""" 
Neural net constructor for random initialisation with dim and K input degree of each hidden unit
"""
function NeuralNetwork(nnDims,K;actF=tanh,varB=0.0)
    NeuralNetwork(RandomInitialization(),nnDims,K;actF=actF,varB=varB)
end
""" 
Neural net constructor given input and output weights
"""
function NeuralNetwork(nnDims,Z,W;actF=tanh,varB=0.0)
    nn = Chain(
        Dense(Z, true, actF),
        Dense(W,false,identity)
    )
    # set biases of the hidden layer
    if varB>0.0
        # nn[1].bias .= rand(Normal(0,varB),nnDims[2])
        # nn[1].bias .= abs.(rand(Normal(0,varB),nnDims[2]))
        # nn[1].bias .= varB*ones(nnDims[2])
        if actF==tanh
            nn[1].bias .= rand(Uniform(-varB,varB),nnDims[2])
        else
            nn[1].bias .= rand(Uniform(0.0,varB),nnDims[2])
        end
    elseif varB<0.0
        if actF==tanh
            nn[1].bias .= rand(Uniform(varB,-varB),nnDims[2])
        else
            nn[1].bias .= rand(Uniform(varB,0.0),nnDims[2])
        end
    end
    mlp = MLP_controller(nn) # make ODE system of the nn
    NeuralNetwork(nnDims,Z,W,nn,mlp)
end

"""
get all weights of the flux nn network
"""
function getAllWeights(nn::NeuralNetwork)
    pAll,re = Flux.destructure(nn.fluxNet)
    return pAll
end
"""
get input weights of the flux nn network
"""
function getInputWeights(sys::System) 
    getInputWeights(sys.nn)
end
function getInputWeights(nn::NeuralNetwork)
    pI,re = Flux.destructure(nn.fluxNet[1:end-1])
    return pI
end
"""
get biases of the flux nn network
"""
function getBiases(nn::NeuralNetwork)
    return nn.fluxNet[1].bias
end
"""
get number of input weights of the neural net
"""
function getNumInputWeigts(nn::NeuralNetwork)
    pI, reInput = Flux.destructure(nn.fluxNet[1:end-1]) # input weights
    length(pI)
end
"""
get number of output weights of the neural net
"""
function getNumOutputWeights(syst::System)
    getNumOutputWeights(syst.nn)
end
"""
get number of output weights of the neural net
"""
function getNumOutputWeights(nn::NeuralNetwork)
    pOut, reOut = Flux.destructure(nn.fluxNet[end:end]) # input weights
    length(pOut)
end
"""
get number of hidden units N of the neural net of a syst
"""
function getN(syst::System)
    getN(syst.nn)
end
"""
get number of hidden units N of the neural net nn
"""
function getN(nn::NeuralNetwork)
    nn.dims[2]
end

""" 
Reference: generates reference trajectory and lookahead vector
"""
struct Reference{A,S,T}
    func::A 
    lookahead_times::S
    system::T
end
""" 
Reference: constructor given the function func(t) gives the reference at time
lookahead_times the array of times to generate the inputs to the neual net
"""
function Reference(func,lookahead_times)
   syst = create_ref(func,lookahead_times)
   Reference(func,lookahead_times,syst) 
end

""" 
Reference: given parameters for O-U process with filtering
lookahead_times the array of times to generate the inputs to the neual net
"""
function Reference(lookahead_times,θ,sigma,μ,T,dt,fc)
    func = ouFunDef(θ,sigma,μ,T,dt,fc)
    syst = create_ref(func,lookahead_times) 
    Reference(func,lookahead_times,syst) 
end 

""" 
Plant linear plant with state space given by plant matrices
"""
struct LTIPlant{A,T} <: Plant
    plantMatrices::A
    system::T
end
function LTIPlant(plantMatrices)
    sys = linear_plant(plantMatrices...)
    LTIPlant(plantMatrices,sys)
end

"""
    Plant of the type RNN with connection matrix J shape (MxM) 
    where M is the number of units in the RNN 
    g is the gain for each unitgain 
    output_index is the unit setting the output
    system: is the ode system of the RNN 
"""
# struct RNNPlant{A,B,D,T} <: Plant 
#     J::A
#     g::B
#     output_index::D
#     system::T
# end
struct RNNPlant{A,B,D,E,T} <: Plant 
    J::A
    g::B
    Jout::D
    tau::E
    system::T
end
# function RNNPlant(J,g,output_index=[])
#     if isempty(output_index) 
#         output_index = size(J,1) # default output is last output 
#     end
#     sys = rnn_plant(J, g, output_index)
#     RNNPlant(J,g,output_index,sys)
# end
function RNNPlant(J,g,Jout,tau)
    sys = rnn_plant(J, g, Jout,tau)
    RNNPlant(J,g,Jout,tau,sys)
end
"""
    constructor if no connection matrix is given 
        only number of units N and variance of J_{ij}
"""
function RNNPlant(N::Int, g, Jvar = 1, tau = 0.01, Jout = [])
    J = randomJ(N,Jvar)
    d = Normal(0,1/sqrt(N))
    if isempty(Jout)
        Jout = rand(d,(1,N))
    end 
    RNNPlant(J,g,Jout,tau)
end

"""
    function to generate random connection matrix between N units
    with no self-conenctivity
"""
function randomJ(N,Jvar)
    J_var = Jvar^2 / N; # variance for weights distribution
    J_mat = sqrt(J_var) * randn(N,N); # weight matrix
    zero_self = 1 .- I(N); 
    J_mat = J_mat .* zero_self; #set self-connectivity to zero
end

""" 
    non-liner 2 joint planar arm plant  
"""
struct NLArmPlant{A,T} <: Plant
    param::A # array with parameters of plant
    system::T
end
function NLArmPlant(param)
    sys = nlArmPlantF(param)
    NLArmPlant(param,sys)
end

""" 
PID controller with gains = (Pgain, Igain, Dgain, RC)
"""
struct PID{A,T} <: FBController
    gains::A
    system::T
end
function PID(gains::Tuple)
   PID(gains, 1)
end
function PID(gains::Tuple,num_inputs::Int64)
    sys =  PID_controller(gains...,num_inputs)
    PID(gains, sys)
end
