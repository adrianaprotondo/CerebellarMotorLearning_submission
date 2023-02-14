module CerebellarMotorLearning

using ModelingToolkit, OrdinaryDiffEq, Flux, DiffEqSensitivity, Symbolics, DiffEqCallbacks
using Distributions
using Plots, LaTeXStrings
using Random, LinearAlgebra
using ForwardDiff
# using ReverseDiff
using DiffResults
# using Zygote
using LsqFit
using Combinatorics 
using Symbolics: scalarize
using DSP, Interpolations, DifferentialEquations # for OrnsteinUhlenbeck process reference function
# using ForwardDiff, DiffEqFlux

include("types.jl")
include("systemComponents_functions.jl")
include("training_extractors.jl")
include("training_Components.jl")
include("trainingComponents_functions.jl")
include("postAnalysis_functions.jl")
include("postAnalysis_components.jl")
include("store_functions.jl")
include("simulation_functions.jl")

export build_system, buildOdeTrainingProblem, build_store, createZ
export sinFunDef, ouFunDef, refFourier
export build_inputSystem, InputSystem
export OnlineGradientTrain, GaussianNoise, LMSTrain, GradientTrain
export OnlineGradientCompute, GaussianNoiseCompute, LMSCompute
export makeCallbacks, solve_prob, train, simulate
export makeCallback_record, makeCallback_update
export extract_solData, update_store!
export LocalTaskDifficulty, GradCorr, LearningPerformanceTrain, LearningPerformanceTask, HessianAnalysis, Hessian, CodingLevel
export buildODETestProb, postProcess, getInputWeights, getN, getAllWeights, getBiases, getNumOutputWeights
export plotSummary
export connectInputSystem
export get_hiddens
export RNNPlant, Plant, NLArmPlant
export NeuralNetwork, PID, Reference, connectFFFB, System
export buildODETestProbF

end
