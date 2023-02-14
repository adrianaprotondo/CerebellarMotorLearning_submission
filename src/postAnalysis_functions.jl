###
# Helper functions for postAnalysis 
###

"""
Function that defines the task loss of ODEproblem prob
pI gives the input weights of the NN (the parameters of the ODE that are static)
trainErrorIndex gives the index in the solution that gives the integral of the error
"""
function makeTaskErrorF(prob,pI,trainErrorIndex,solver=Tsit5())
    function G(ps)
        tmp_prob = remake(prob,p=vcat(pI,ps))
        # tmp_prob = remake(prob,p=pnew)
        # sol = solve(tmp_prob,Euler(),dt=0.001)
        sol = solve(tmp_prob,solver)
        T = sol.t[end]
        sol[trainErrorIndex,end]/T # normalize by length of trajectory
    end
end
"""
return task loss for output weights pOutput, input weights pI
ODE problem prob
"""
function getTaskError(pOutput,pI,prob,trainErrorIndex,solver=Tsit5())
    # return task error of the system with weights pOutput
    G = makeTaskErrorF(prob,pI,trainErrorIndex,solver) 
    taskError = G(pOutput)
    return taskError
end

"""
Get gradient of task error (error over whole traj) 
    with forward differentiation
    return the task error value and gradient
"""
function getGradLoss(pOutput,pI,prob,trainErrorIndex,solver=Tsit5())
    pI = prob.p[1:end-length(pOutput)]
    G = makeTaskErrorF(prob,pI,trainErrorIndex,solver)
    taskError = G(pOutput)
    # gradFinal = ForwardDiff.gradient(G,pOutput)
    gradFinal = similar(pOutput)
    ForwardDiff.gradient!(gradFinal,G,pOutput)
    return taskError,gradFinal
end

"""
Get gradient of task error and hessian (error over whole traj) 
with forward differentiation
return the task error value and gradient
"""
function getGradAndHess(pOutput,pI,prob,trainErrorIndex,solver=Tsit5())
    G = makeTaskErrorF(prob,pI,trainErrorIndex,solver)
    result = DiffResults.HessianResult(pOutput)
    println("Computing grad and hessian")
    result = ForwardDiff.hessian!(result, G, pOutput); 
    return DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
end

"""
    Compute projection of vector onlineGrad into hessian of task error at pOutput
    v̂ᵀ∇²F[w]v̂
"""
function gradHessProj(pOutput,onlineGrad,prob,pI,trainErrorIndex,solver=Tsit5())
    # getting hessian projections without calculating the hessian
    nv = normalize(onlineGrad) # normalize vector
    G = makeTaskErrorF(prob,pI,trainErrorIndex,solver) # generate function of task error
    # function gradDotP(pOutput,v)
    #     gradFinal = ForwardDiff.gradient(G,pOutput)
    #     return v'*gradFinal
    #     # return prod
    # end
    # prod = ForwardDiff.gradient(w -> gradDotP(w, nv), pOutput)
    # return prod'*nv
    # _w = ForwardDiff.Dual.(pOutput, nv)
    # rr = Zygote.gradient(G, _w)[1]
    # """ Hu is the product of the Hessian of the loss function with v"""
    # Hv = getindex.(ForwardDiff.partials.(rr), 1)
    # return  v'*Hv
    function autoback_hesvec(f, x, v)
        # g = x -> first(Zygote.gradient(f,x))
        g = x -> ForwardDiff.gradient(f,x)
        ForwardDiff.partials.(g(ForwardDiff.Dual{Nothing}.(x, v)), 1)
    end
    Hv = autoback_hesvec(G,pOutput,nv)
    # println(Hv)
    # println(size(Hv))
    return nv'*Hv
    # return proj
end

"""
Fit task loss with exponential descent to compute learning speed and steady state loss
"""
@. model(x, p) = p[1]*exp(-x*p[2])+p[3]
function fitLoss(loss,times,m="exp")
    if m == "exp"
        @. model(x, p) = p[1]*exp(-x*p[2])+p[3]
    end
    p0 = [loss[1],0.1,loss[end]]
    lb = [0.0, 0.0, 0.0000000000001]
    ub = [Inf, Inf, 10*loss[1]]
    try 
        fit = curve_fit(model,times,loss, p0, lower=lb, upper=ub)
        p = fit.param
        return p
    catch 
        println("problem with fit")
        p = [loss[1],NaN,mean(loss[end-2:end])]
        return p
    end
end