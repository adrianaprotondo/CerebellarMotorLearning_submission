############
# Theoretical curves 
# change in weights with normalised gradO 
# loads learning parameters from simulation
# plot ls and local task difficulty varying
# q, gamma, gammaO, SNR 
############
using Measures

include("default.jl")
include("sizeSim_functions.jl")

path = string("theoreticalCurves_normalisedGradO")
mkpath(datadir("simulations", path))
mkpath(plotsdir(path))


######## load data from sim to get experimental values
simModel = "test_assumptions"
loadPath = string(simModel,"_",10,"_",110)

df = collect_results(datadir("simulations",loadPath);verbose=true) # get data in the path with simulations

# values of parameters used for simulations
seedsA = unique(df[!,:seed])
mus = unique(df[!,:mu])[1]
t_save = df[!,:t_save][1]

Ns = unique(df[!,:Ns])[1] # get net sizes
NsNorm = Ns./Ns[1] # expansion ratio
Nmus = length(mus[1]) # number of different mus for each size
postStores = map(seedsA) do s # get postStore for each net size for a given seed
    filter(row -> (row.seed==s),df)[:,:postStores][1]
end

initialGrad = map(postStores) do p 
    mean(p[1][:norm_grad])
end
endGrad = map(postStores) do p 
    mean(p[1][:norm_grad][end-50:end])
end
initialGradO = map(postStores) do p 
    mean(p[1][:norm_gradO][1:100])
end

initialHessProj = map(postStores) do p 
    mean(p[1][:hessProj_gradO])
end 
initialTr_N = map(postStores) do p 
    minimum(p[1][:hessProj_gradO][1,2:end])
end 

"""
    rho as a function of gammaO (the grad projection)
"""
function rho_gammaO(gammaO,c_rho=3)
    abs(2/(1+exp(-c_rho*gammaO))-1)
end

"""
    compute the hessian projection of the online gradient in the expand net
    given values in the small net, expansion factor k and scaling rho
    the proejction is a linear combination of the tr3_tr2 term growing with k^{rho}
    and tr_N term that stays constant 
    hat{∇L^T} H hat{∇L} = c_1(γ_o) Tr(H^3)/Tr(H^2) *k^ρ + c_2(γ_o) *Tr(H)/N
"""
function gradOHessProjExp(tr3_tr2,tr_N,gammaO,rho,k)
    # gammaO^2*tr3_tr2*k^rho + (1-gammaO^2)*tr_N
    # tr3_tr2*k^rho*gradOTr3_tr2(gammaO)+gradOTr_N(gammaO)*tr_N
    # tr3_tr2*k^rho
    tr3_tr2*gammaO^2*k^rho_gammaO(gammaO)
end

""" Expansion of task gradient """
function gradExp(grad,k,a=0.5)
    grad*k^a
end

""" Expansion of online gradient"""
function gradOExp(gradO,k,a=0.5)
    gradO*k^a
end

""" Expansion of hessian projection of learning rule error"""
function lrErrProj(tr_N,k,a=0)
    tr_N*k^a
end

""" 
    function to compute the learning speed in the expanded net
    given values in the small net and expansion ratio k 
    ls = -(-γγ_o*||∇F||*√k + 1/2*γ^2 expanded(hat{∇L}^T H hat{∇L})+ 1/2*η^lr^2*Tr(H)/N)
"""
function lsExp(grad,tr3_tr2,tr_N,gamma,etaLr,gammaO,rho,k,gradO=0.3)
    return -(-gamma*gammaO*gradExp(grad,k)+1/2*gamma^2*gradOHessProjExp(tr3_tr2,tr_N,gammaO,rho,k)
        +1/2*etaLr^2*lrErrProj(tr_N,k))
    # return -(-gamma*gammaO*gradExp(grad,k)*gradOExp(gradO,k)+1/2*gamma^2*gradOExp(gradO,k)^2*gradOHessProjExp(tr3_tr2,tr_N,gammaO,rho,k)
        # +1/2*etaLr^2*lrErrProj(tr_N,k))
end


# function kOptLs(grad,tr3_tr2,tr_N,gamma,etaLr,gammaO,rho)
#     return (gamma*gammaO*grad/(rho*gamma^2*gradOTr3_tr2(gammaO)*tr3_tr2))^(1/(rho-0.5))
# end

""" 
    function to compute the local task diff in the expanded net
    given values in the small net and expansion ratio k 
    G = (γ^2 expanded(hat{∇L}^T H hat{∇L})*||∇L||*√k + 1/2*η^lr^2*Tr(H)/N*1/(||∇L||*√k)))/(||∇F||*√k)
"""
function localTaskDExp(grad,tr3_tr2,tr_N,gamma,etaLr,gammaO,rho,k,gradO=0.3,Lss0=0.0)
    # ltD = gamma*(gradOExp(gradO,k)/gradExp(grad,k))*gradOHessProjExp(tr3_tr2,tr_N,gammaO,rho,k)+(etaLr^2/gamma)*lrErrProj(tr_N,k)/(gradOExp(gradO,k)*gradExp(grad,k))
    ltD = 1/2*gamma/gradExp(grad,k)*gradOHessProjExp(tr3_tr2,tr_N,gammaO,rho,k)+1/2*(etaLr^2/gamma)*lrErrProj(tr_N,k)/(gradExp(grad,k))
    return ltD + Lss0/k
end

"""
    compute ls and ss over ks and gammasVar 
    gammasVar has shape length(ks)xnumberGammas
"""
function computeLS_SS(ks,gammasVar,SNR,gammaO,gammaOSS)
    ls = map(1:length(ks)) do i
        k = ks[i]
        gs = gammasVar[i]
        map(gs) do g
            SNRConstant==true ? etaLr = g/SNR : etaLr = etaLrF
            lsExp(gradNorm,tr3_tr2,tr_N,g,etaLr,gammaO,rho,k,gradO)
        end
    end
    
    ss = map(1:length(ks)) do i
        k = ks[i]
        gs = gammasVar[i]
        map(gs) do g
            SNRConstant==true ? etaLr = g/SNR : etaLr = etaLrF
            localTaskDExp(gradNormSS,tr3_tr2,tr_N,g,etaLr,gammaOSS,rho,k,gradO)
        end
    end
    return ls, ss
end

function optimalLS_SS(lsN,ssN,gammasVar)

    maxLs = map(lsN) do x
        maximum(x)
    end 
    gammaOptLs = map(1:length(lsN)) do j
        x = lsN[j]
        i = argmax(x)
        gammasVar[j][i]
    end 
    # ss at the max ls
    maxSs = map(1:length(lsN)) do j
        i = argmax(lsN[j])
        ssN[j][i]
    end 
    return maxLs,maxSs, gammaOptLs
end

# function kOptg(grad,tr3_tr2,tr_N,gamma,etaLr,gammaO,rho)
#     return ((gamma^2*gradOTr_N(gammaO)*tr_N+etaLr^2*tr_N)/(2*(rho-1/2)*gamma^2*gradOTr3_tr2(gammaO)*tr3_tr2))^(1/rho)
# end

gamma = 0.05 # decreases koptLs
SNR = 3
etaLr = gamma/SNR # 
gammaO = 0.7
gammaOSS = 0.2
# gradNorm = 0.1
gradNorm = mean(initialGrad)
# gradNormSS = 0.05
gradNormSS = mean(endGrad)
# gradO = 0.1
gradO = mean(initialGradO)
# tr3_tr2= 1
tr3_tr2 = mean(initialHessProj)
# tr_N = 0.05
tr_N = mean(initialTr_N)
rho = 0.9

ks = 1:1:20

##########
# Trade-off with gammas. Scale gamma with net expansion 
###########
gammas = 0.01:0.01:0.3
gammasVar = [gammas./k^0.5 for k in ks]
SNR = 3
SNRConstant = true # change eta for different gamma to keep SNR

cm = palette(:heat,length(ks)+1)[2:end]
lbl = string.(L"q=",ks)
lbl = reshape(lbl,(1,length(lbl)))

ls_gammas2,ss_gammas2 = computeLS_SS(ks,gammasVar,SNR,gammaO,gammaOSS) 

# plot_mean(gammasVar,[ls_gammas2],lbl,L"\gamma","Learning speed",plotsdir(path,"ls_k_gammas_2.pdf"),false,:line,cm)
plot(gammasVar,ls_gammas2,lw=3,label=lbl,legend=:outertopright,
    xlabel=L"\gamma",ylabel="Learning speed",palette=cm)
savefig(plotsdir(path,"ls_k_gammas_2_gammaO-$gammaO.pdf"))
# plot_mean(gammasVar,[ss_gammas2],lbl,L"\gamma","Steady State error",plotsdir(path,"ss_k_gammas_2.pdf"),false,:line,cm)
plot(gammasVar,ss_gammas2,lw=3,label=lbl,legend=:outertopright,
    xlabel=L"\gamma",ylabel="Steady state loss",palette=cm)
savefig(plotsdir(path,"ss_k_gammas_2_gammaO-$gammaOSS.pdf"))
# plot(gammasVar,ss_gammas2)

# heatmap(Ns,Ks,varsMM', xlabel=xlbl, ylabel=leglbl,title=savelbl)


#########
### Scatter plot
#########
# int = 2:10:length(ks)
int = 2:2:10
# int = 2:1:2
ss = [ss_gammas2[int]]
ssN = ss./ss_gammas2[1][1]
ls = [ls_gammas2[int]]
lsN = ls./ls_gammas2[1][1]
# ssPerf = map(ss) do x
#     map(x) do y
#         1 .-y
#     end
# end
l = lbl[int]
plot_scatterMean(ss,ls,l,1:length(gammas),plotsdir(path,"ss_vs_ls_gammas_gammaO-$gammaO.pdf"),false,6)
# plot_scatterMean(ss,ls,l,5:length(gammas),plotsdir(path,"ss_vs_ls_gammas.pdf"))
plot_scatterMean(ssN,lsN,l,1:length(gammas),plotsdir(path,"ss_vs_ls_N_gammas_gammaO-$gammaO.pdf"),false,6) 


int = 2:4:20
ss = [ss_gammas2[int]]
ls = [ls_gammas2[int]]
ssN = ss./ss_gammas2[1][1]
lsN = ls./ls_gammas2[1][1]
l = lbl[int]
plot_scatterMean(ss,ls,l,1:length(gammas),plotsdir(path,"ss_vs_ls_gammas_2_gammaO-$gammaO.pdf"),false,6)
plot_scatterMean(ssN,lsN,l,1:length(gammas),plotsdir(path,"ss_vs_ls_N_gammas_2_gammaO-$gammaO.pdf"),false,6) 

int = 2:1:2
ss = [ss_gammas2[int]]
ls = [ls_gammas2[int]]
ssN = ss./ss_gammas2[1][1]
lsN = ls./ls_gammas2[1][1]
l = lbl[int]
plot_scatterMean(ss,ls,l,1:length(gammas),plotsdir(path,"ss_vs_ls_gammas_1_gammaO-$gammaO.pdf"),false,6)
plot_scatterMean(ssN,lsN,l,1:length(gammas),plotsdir(path,"ss_vs_ls_N_gammas_1_gammaO-$gammaO.pdf"),false,6) 

####
# optimal Ls and SS
####
ls_gammas2_N = [ls_gammas2[i]./ls_gammas2[1][1] for i=1:length(ls_gammas2)]
ss_gammas2_N = [ss_gammas2[i]./ss_gammas2[1][1] for i=1:length(ss_gammas2)]

maxLs,maxSs,gammaOptLs = optimalLS_SS(ls_gammas2_N,ss_gammas2_N,gammasVar)  

## Plot optimal ls and ss as a function of k 
# and gamma^* (gamma such that ls is optimal) as a function of k 
plot_mean(ks,[maxLs],[L"\nu^*"],L"k","Max Learning Speed",plotsdir(path,"lsMax_gamma.pdf"),false,:line)
plot_mean(ks,[gammaOptLs],[L"\gamma^*_{\nu}"],L"k",L"\gamma",plotsdir(path,"gammaOpt_LS.pdf"),false,:line)
plot_mean(ks,[maxSs],[L"\xi"],L"k",L"steady state loss at $\nu^*$",plotsdir(path,"SsMin_gamma.pdf"),false,:line)

plot(ks,maxLs,xlabel="expansion ratio (q)",ylabel=L"optimal learning speed ($\nu^*$)",grid=false,lw=3,legend=:false,color=:blue)
plot!(twinx(),ks,maxSs,axis=:right,ylabel="local task difficulty",lw=3,legend=false,color=:orange)
plot!(right_margin=20mm)
savefig(plotsdir(path,"ls_and_SsMin_gamma.pdf"))

# gammaOptSs = map(ss_gammas2) do x
#     i = argmin(x)
#     gammas[i]
# end 
# plot_mean(ks,[gammaOptSs],[L"\gamma^*_{\xi}"],L"k",L"\gamma",plotsdir(path,"gammaOpt_SS.pdf"),false,:line)

#######
# Low versus high gamma
#######

gammaO1 = 0.99
gammaO2 = 0.7
gammaOSS1 = 0.6
gammaOSS2 = 0.3

gammas = 0.01:0.01:0.3
gammasVar = [gammas./k^0.5 for k in ks]
SNR = 3
SNRConstant = true # change eta for different gamma to keep SNR

ls1,ss1 = computeLS_SS(ks,gammasVar,SNR,gammaO1,gammaOSS1) 
# lsN = [ls[i]./ls[1][1] for i=1:length(ls)]
# ssN = [ss[i]./ss[1][1] for i=1:length(ss)]
maxLs1,maxSS1,gammaOpt1 = optimalLS_SS(ls1,ss1,gammasVar) 

ls2,ss2 = computeLS_SS(ks,gammasVar,SNR,gammaO2,gammaOSS2) 
maxLs2,maxSS2,gammaOpt2 = optimalLS_SS(ls2,ss2,gammasVar) 

plot(ks,maxLs1,label=string(L"\gamma^o=",gammaO1),lw=3)
plot!(ks,maxLs2,label=string(L"\gamma^o=",gammaO2))

plot(ks,maxSS1,label=string(L"\gamma^o=",gammaO1))
plot!(ks,maxSS2,label=string(L"\gamma^o=",gammaO2))

####
# Vary gammaO 
####
gamma = 0.08
gammaOs = collect(0.5:0.05:1.0)
cm = palette(:heat,length(gammaOs)+1)[2:end]
ls_gammaOs = map(gammaOs) do gO
    map(1:length(ks)) do i
        k = ks[i]
        lsExp(gradNorm,tr3_tr2,tr_N,gamma,etaLr,gO,rho,k,gradO)
    end
end
lbl = string.(L"\gamma^{o}=",gammaOs)
lbl = reshape(lbl,(1,length(lbl)))
lsN = [ls_gammaOs[i]./ls_gammaOs[1][1] for i=1:length(ls_gammaOs)]
plot_mean(ks,[lsN],lbl,"input expansion","learning speed",plotsdir(path,"lsN_k_gammaOs_gamma_$gamma.pdf"),false,:line,cm)
# plot_mean(ks,[ls_gammaOs],lbl,"input expansion","learning speed",plotsdir(path,"ls_k_gammaOs_gamma_$gamma.pdf"),false,:line,cm)


gamma=0.002
# gammaOsSS = 0.1:0.05:0.6
gammaOsSS = collect(0.05:0.05:0.6)
cm = palette(:heat,length(gammaOsSS)+1)[2:end]
ss_gammaOs = map(gammaOsSS) do gO
    map(ks) do k
        localTaskDExp(gradNormSS,tr3_tr2,tr_N,gamma,etaLr,gO,rho,k,gradO)
    end
end
lbl = string.(L"\gamma^{o}=",gammaOsSS)
lbl = reshape(lbl,(1,length(lbl)))
ssN = [ss_gammaOs[i]./ss_gammaOs[1][1] for i=1:length(ss_gammaOs)]
plot_mean(ks,[ssN],lbl,"input expansion","steady state loss",plotsdir(path,"ssN_k_gammaOs_gamma_$gamma.pdf"),false,:line,cm)
plot_mean(ks,[ss_gammaOs],lbl,"input expansion","steady state loss",plotsdir(path,"ss_k_gammaOs_gamma_$gamma.pdf"),false,:line,cm)


cm = palette(:heat,length(ks)+1)[2:end]
ls_gammaOs2= map(ks) do k
    map(gammaOs) do g    
        lsExp(gradNorm,tr3_tr2,tr_N,gamma,etaLr,g,rho,k,gradO)
    end
end
lbl = string.(L"q=",ks)
lbl = reshape(lbl,(1,length(lbl)))
plot_mean(gammaOs,[ls_gammaOs2],lbl,L"\gamma^{o}","Learning speed",plotsdir(path,"ls_k_gammaO_2.pdf"),false,:line,cm)

### Get hessian projections for different gammas
gammaOs = reverse(collect(0.0:0.05:1.0))

cm = palette(:heat,length(gammaOs)+1)[2:end]
gradOProjs = map(gammaOs) do g
    map(ks) do k
        gradOHessProjExp(tr3_tr2,tr_N,g,rho,k)
        # localTaskDExp(gradNorm,tr3_tr2,tr_N,g,etaLr,gammaOSS,rho,k,gradO)
    end
end
lbl = string.(L"\gamma^{o}=",gammaOs)
lbl = reshape(lbl,(1,length(lbl)))
plot_mean(ks,[gradOProjs],lbl,"Input expansion","gradO hess proj",plotsdir(path,"gradOHessProj_gammaOs.pdf"),false,:line,cm)

cm = palette(:heat,length(ks)+1)[2:end]
gradOProjs = map(ks) do k
    map(gammaOs) do g
        gradOHessProjExp(tr3_tr2,tr_N,g,rho,k)
    end
end
lbl = string.(L"q=",ks)
lbl = reshape(lbl,(1,length(lbl)))
plot_mean(gammaOs,[gradOProjs],lbl,L"\gamma^{o}","gradO hess proj",plotsdir(path,"gradOHessProj_k_gammaOs.pdf"),false,:line,cm)


# gammaOsSS = 0.1:0.1:0.6
ss_gammaOs2 = map(ks) do k
    map(gammaOsSS) do g
        localTaskDExp(gradNormSS,tr3_tr2,tr_N,gamma,etaLr,g,rho,k,gradO)
    end
end
plot_mean(gammaOsSS,[ss_gammaOs2],lbl,L"\gamma^{o}","Steady State error",plotsdir(path,"ss_k_gammaO_2.pdf"),false,:line,cm)

int = 2:10:length(ks)
ss = [ss_gammaOs2[int]]
ls = [ls_gammaOs2[int]]
# ssPerf = map(ss) do x
#     map(x) do y
#         1 .-y
#     end
# end
l = lbl[int]
plot_scatterMean(ss,ls,l,1:length(gammaOs),plotsdir(path,"ss_vs_ls_gammaOs.pdf"))
# plot_scatterMean(ssPerf,ls,l,1:length(gammaOs),plotsdir(path,"ssP_vs_ls_gammaOs.pdf")) 

#######
# Vary SNR (learning rule error)
##########
gamma = 0.01
SNRs = 1:0.5:10
SNRConstant = true # change eta for different gamma to keep SNR
gammaO = 0.5
gammaOSS= 0.5

lbl = string.(L"q=",ks)
lbl = reshape(lbl,(1,length(lbl)))

ls_SNR = map(SNRs) do SNR
    etaLr = gamma/SNR 
    map(ks) do k
        lsExp(gradNorm,tr3_tr2,tr_N,gamma,etaLr,gammaO,rho,k,gradO)
    end
end

ss_SNR = map(SNRs) do SNR
    etaLr=gamma/SNR
    map(ks) do k
        localTaskDExp(gradNormSS,tr3_tr2,tr_N,gamma,etaLr,gammaOSS,rho,k,gradO)
    end
end


cm = palette(:heat,length(SNRs)+1)[2:end]
lbl = string.(L"\sigma=",SNRs)
lbl = reshape(lbl,(1,length(lbl)))
ls_SNR_N = [ls_SNR[i]./ls_SNR[1][1] for i=1:length(ls_SNR)] 
plot_mean(ks,[ls_SNR_N],lbl,"input expansion","learning speed",plotsdir(path,"lsN_k_snr_gamma_$gamma.pdf"),false,:line,cm)

ss_SNR_N = [ss_SNR[i]./ss_SNR[1][1] for i=1:length(ss_SNR)] 
plot_mean(ks,[ss_SNR_N],lbl,"input expansion","local task difficulty",plotsdir(path,"ssN_k_snr_gamma_$gamma.pdf"),false,:line,cm)




####
# Vary etaLr
####
# gradNorm = 0.3
# ks = 1:1:20
# rho= 0.8

etaLrs =gamma/10:gamma/10:gamma
cm = palette(:heat,length(etaLrs)+1)[2:end]
ls_etaLrs= map(etaLrs) do e
    map(ks) do k
        lsExp(gradNorm,tr3_tr2,tr_N,gamma,e,gammaO,rho,k,gradO)
    end
end
lbl = string.(L"\eta^{lr}=",etaLrs)
lbl = reshape(lbl,(1,length(lbl)))
plot_mean(ks,[ls_etaLrs],lbl,"Input expansion","Learning speed",plotsdir(path,"ls_k_etaLrs.pdf"),false,:line,cm)

# gammaOsSS = 0.1:0.1:0.6
ss_etaLrs = map(etaLrs) do e
    map(ks) do k
        localTaskDExp(gradNormSS,tr3_tr2,tr_N,gamma,e,gammaOSS,rho,k,gradO)
    end
end
plot_mean(ks,[ss_etaLrs],lbl,"Input expansion","Steady State error",plotsdir(path,"ss_k_etaLrs.pdf"),false,:line,cm)



cm = palette(:heat,length(ks)+1)[2:end]
ls_etaLrs2= map(ks) do k
    map(etaLrs) do g    
        lsExp(gradNorm,tr3_tr2,tr_N,gamma,g,gammaO,rho,k,gradO)
    end
end
lbl = string.(L"q=",ks)
lbl = reshape(lbl,(1,length(lbl)))
plot_mean(etaLrs,[ls_etaLrs2],lbl,L"\eta_{lr}","Learning speed",plotsdir(path,"ls_k_etaLr_2.pdf"),false,:line,cm)

# gammaOsSS = 0.1:0.1:0.6
ss_etaLrs2 = map(ks) do k
    map(etaLrs) do g
        localTaskDExp(gradNormSS,tr3_tr2,tr_N,gamma,g,gammaOSS,rho,k,gradO)
    end
end
plot_mean(etaLrs,[ss_etaLrs2],lbl,L"\eta_{lr}","Steady State error",plotsdir(path,"ss_k_etaLr_2.pdf"),false,:line,cm)

int = 2:10:length(ks)
ss = [ss_etaLrs2[int]]
ls = [ls_etaLrs2[int]]
# ssPerf = map(ss) do x
#     map(x) do y
#         1 .-y
#     end
# end
l = lbl[int]
plot_scatterMean(ss,ls,l,1:length(etaLrs),plotsdir(path,"ss_vs_ls_etaLrs.pdf"))
# plot_scatterMean(ssPerf,ls,l,1:length(etaLrs),plotsdir(path,"ssP_vs_ls_etaLrs.pdf")) 



####
# Vary gamma
####
# gradNorm = 0.3
# ks = 1:1:20
# rho= 0.8

gammas = 0.002:0.01:0.15
# etaLr = gammas[1]/2
etaLrF = 0.2
# etaLr = 0
SNR = 3
SNRConstant = true # change eta for different gamma to keep SNR

## GET LS for different Gammas
cm = palette(:heat,length(gammas)+1)[2:end]
ls_gammas= map(gammas) do g
    map(ks) do k
        SNRConstant==true ? etaLr = g/SNR : etaLr = etaLrF
        lsExp(gradNorm,tr3_tr2,tr_N,g,etaLr,gammaO,rho,k,gradO)
    end
end
lbl = string.(L"\gamma=",gammas)
lbl = reshape(lbl,(1,length(lbl)))
plot_mean(ks,[ls_gammas],lbl,"Input expansion","Learning speed",plotsdir(path,"ls_k_gammas.pdf"),false,:line,cm)

## GET SS for different Gammas
ss_gammas = map(gammas) do g
    map(ks) do k
        SNRConstant==true ? etaLr = g/SNR : etaLr = etaLrF
        localTaskDExp(gradNormSS,tr3_tr2,tr_N,g,etaLr,gammaOSS,rho,k,gradO)
    end
end
plot_mean(ks,[ss_gammas],lbl,"Input expansion","Steady State error",plotsdir(path,"ss_k_gammas.pdf"),false,:line,cm)

### Get hessian projections for different gammas
gradOProjs = map(gammas) do g
    map(ks) do k
        SNRConstant==true ? etaLr = g/SNR : etaLr = etaLrF
        gradOHessProjExp(tr3_tr2,tr_N,gammaOSS,rho,k)
        # localTaskDExp(gradNorm,tr3_tr2,tr_N,g,etaLr,gammaOSS,rho,k,gradO)
    end
end
plot_mean(ks,[gradOProjs],lbl,"Input expansion","gradO hess proj",plotsdir(path,"gradOHessProj_gammas.pdf"),false,:line,cm)
