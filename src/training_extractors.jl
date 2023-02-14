###
# Helper functions for training components. Functions to compute the online gradients and error information 
# for training online
###

###
# For online gradient descent
###
""" 
Online gradient extract to save gradients at te_times and tr_times to generate online gradient to train 
"""
struct OnlineGradientExtract{C,D} <: InSimulationOperations 
    te_times::C 
    tr_times::D
end
"""
initialise OnlineGradientExtract with deltatTe, deltaTr and t_train
defines the times at which the ODEsolver must stop to compute for online gradient computation
"""
function OnlineGradientExtract(deltaTe,deltaTr,t_train)
    t_obs1 = t_train .- deltaTe
    te_times = vcat(t_train[1],t_obs1[2:end]) # start of online error window
    t_obs2 = t_train .- deltaTr
    tr_times = vcat(t_train[1],t_obs2[2:end]) # end of online error window
    OnlineGradientExtract(te_times,tr_times)
end

"""
call online gradient extract to make the callbacks to track the gradients at te_time and tr_times
"""
function (o::OnlineGradientExtract)(sys::System,store)
    gradExtF = makeGradExtractor(sys) # function that extracts gradient
    cbTe = makeCallback_record(o.te_times,[gradExtF],[:valTe],store) # callback at t-Δt_e
    # fnsTr, namesTr = makeFTr(sys)
    # cbTr = makeCallback_record(o.tr_times,fnsTr,namesTr,store)  # callback at t-Δt_r
    cbTr = makeCallback_record(o.tr_times,[gradExtF],[:valTr],store)  # callback at t-Δt_r
    return cbTe,cbTr
end

"""
function to make the functions and names for storing at tr_times
    store gradient of onlineErrorIndex, 
    store instanteneous error e
    and instanteneous grad d/dw(e^2) = e*dy/dw
"""
function makeFTr(sys)
    gradExtF = makeGradExtractor(sys) # gradient of the online error index
    r = sys.ref.func
    outputIndex = sys.outputIndex
    function instErrorExtF(int)
        # global outputIndex
        int.u[outputIndex] - r.(int.t) # instanteneous error
        # int.u[]
    end
    function instGradExtF(int)
        # global outputIndex
        instError = int.u[outputIndex] - r.(int.t) # instanteneous error
        dydw = makeGradExtractor(sys,outputIndex)(int) # gradient of the plant output
        instError.*dydw # online instanteneous gradient
    end
    return [gradExtF,instErrorExtF,instGradExtF],[:valTr,:instETr,:gradInstTr]
end


###
# For LMS
###

""" 
extract integral of trajectory error to save gradients at te_times and tr_times to generate LMS to train 
"""
struct TrajErrorExtract{C,D,E,A} <: InSimulationOperations 
    te_times::C 
    tr_times::D
    th_times::E
    trajErrorIndex::A # index in ode of int e(t) (called eI)
end

function TrajErrorExtract(deltaTe,deltaTr,deltaTh,t_train,trajErrorIndex)
    t_obs1 = t_train .- deltaTe
    te_times = vcat(t_train[1],t_obs1[2:end]) # start of online error window
    # println(te_times)
    t_obs2 = t_train .- deltaTr
    tr_times = vcat(t_train[1],t_obs2[2:end]) # end of online error window
    # println(tr_times)
    t_obs3 = t_train .- deltaTh
    th_times = vcat(t_train[1],t_obs3[2:end]) # end of online error window
    # println(th_times)
    TrajErrorExtract(te_times,tr_times,th_times,trajErrorIndex)
end

"""
call traj error extract to make the callbacks to track the error at te_time and tr_times
"""
function (o::TrajErrorExtract)(sys::System,store)
    extF = makeTrajErrorExtractor(sys,o.trajErrorIndex)
    hiddenF = makeHiddLExtractor(sys)
    cbTe = makeCallback_record(o.te_times,[extF],[:trajETe],store) # callback at t-Δt_e
    cbTr = makeCallback_record(o.tr_times,[extF],[:trajETr],store)  # callback at t-Δt_r
    cbTh = makeCallback_record(o.th_times,[hiddenF],[:hTh],store)  # callback at t-Δt_h
    return cbTe,cbTr,cbTh
end
