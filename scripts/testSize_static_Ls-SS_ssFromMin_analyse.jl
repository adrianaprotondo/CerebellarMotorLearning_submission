#####
# Produce graphs of learning speed and steady state loss 
# for different input expansions Ns and learning steps
# We test two different ways of computing the learning parameters
# 1. Fit the task loss as an exponential decay c_1 e^(-c_2 t )+c_3. 
#       the learning speed is c_2 and steady state loss is c_3
# 2. Learning speed is the mean over first t_ls epochs of νₜ = -ΔFₜ/(δₜFₜ) (where F is the task loss)
#       Steady state loss is the mean of the last task loss over the last t_ls
# Use data produced by testSize_static_Ls-SS_simulate saved in datadir(path)
# Set simModel, path and pretrain used in testSize_static_Ls-SS_simulate 
# Plots produced:
# - optimal learning speed and optimal ss vs Ns and for different gammas
# - scatter ls vs SS for different Ns 
# - line plots ls and ss as a function of gammas or Ns (different color lines)
#####

using CerebellarMotorLearning
using Distributions
using Plots
using Random
using LaTeXStrings
using LinearAlgebra
using LsqFit
using Printf
using DrWatson

using DataFrames
using StatsPlots
using Measures

# Plots.default(titlefont = (20, "times"), legendfontsize = 18, guidefont = (18, :black), tickfont = (12, :black), grid=false, framestyle = :zerolines, yminorgrid = false, markersize=6)
Plots.default( grid=false, markersize=6,linewidth=3)
pwidth=1000
pheight=600

include("sizeSim_functions.jl")
# simModel = "LMS_size_test_seed9"
simModel = "lms_size_Ls-ss_fourrier_o-u_2"
path = string(simModel,"_",10,"_",90)
preTrain = true

############
# Load simulation data as a Dataframe using DrWatson
############
# dfLsSS has a columns named :postStores, mu, Ns,seed
dfLsSS = collect_results(datadir("simulations",path);verbose=true) # get data in the path with simulations
sort!(dfLsSS,:seed)

# values of parameters used for simulations
seeds = unique(dfLsSS[!,:seed])
mus = unique(dfLsSS[!,:mu])[1]
t_save = dfLsSS[!,:t_save][1]

Ns = unique(dfLsSS[!,:Ns])[1] # get net sizes
NsNorm = Ns./Ns[1] # expansion ratio
Nmus = length(mus[1]) # number of different mus for each size


############
## Extract poststores and trainstores according to seed
############
# get postStores an array of arrays size length(seeds)xlength(Ns)xlength(mus)

################### Select Ns ####################
# NsInt = 4:length(Ns)
NsInt = 1:length(Ns)
Ns = Ns[NsInt]
NsNorm = NsNorm[NsInt] # expansion ratio
mus = mus[NsInt]

# get all poststores
postStores = map(seeds) do s # get postStore for each net size for a given seed
    filter(row -> (row.seed==s),dfLsSS)[:,:postStores][1][NsInt]
end

# initial loss (before learning) for each seed (equal for all sizes by design)
initialLoss = [postStores[i][1][1][:taskError][1,1] for i=1:length(seeds)]
plot(unique(dfLsSS[!,:seed]),initialLoss)

if preTrain # get post-stores to compute SS loss if there was pretrain    
    trajTimeSS = 1000.; # traj time for ss calculation
    t_saveSS = t_save[1]:5:trajTimeSS
    postStoresSS = map(seeds) do s # get postStore for each net size for a given seed
        filter(row -> (row.seed==s),dfLsSS)[:,:postStoresSS][1][NsInt]
    end
    loop_3(postStoresSS,computeDynamicSS!,[50,false]) # compute dynamic ss
    # compute ss from normalised task loss
    map(1:length(postStoresSS)) do i 
        loop_2(postStoresSS[i],computeNormalisedTaskE!,[initialLoss[i]])
    end
    # ss is mean of task loss over the second half of traj
    loop_3(postStoresSS,computeDynamicSS!,[Int(length(t_saveSS)/2),false,:taskErrorN,:ssN])
end

# Compute learning speed and steady state loss from mean (described above)
loop_3(postStores,computeDynamicLs!,[t_save,100]) # compute dynamic ls 
loop_3(postStores,computeDynamicSS!,[50]) # compute dynamic ss

# compute normalised task error and learning performance with respect to the normalised value
loop_3(postStores,computeNormalisedTaskE!,[]) # compute dynamic ls 
loop_3(postStores,computeLsSSFit!,[t_save,:taskErrorN]) # compute dynamic ls 


# loss = postStores[10][1][1][:taskError]
# @. model(x, p) = p[1]*exp(-x*p[2])+p[3]
# p0 = [loss[1],0.1,loss[end]]
# lb = [0.0, 0.0, loss[end]/1000]
# ub = [Inf, Inf, 1000*mean(loss)]

# fit = curve_fit(model,t_save,loss[1,:], p0, lower=lb, upper=ub)

# plot(t_saveSS,loss[1,:])
# plot!(t_saveSS,model(t_saveSS,fit.param))

##### Select only postStores with large enough initial loss
# threshold = 0.001
threshold = mean(initialLoss)-std(initialLoss)/2
postStores = postStores[initialLoss.>threshold]
postStoresSS = postStoresSS[initialLoss.>threshold]
seeds = seeds[initialLoss.>threshold]

plot(unique(dfLsSS[!,:seed]),initialLoss,label="initial loss")
plot!([threshold],seriestype=:hline,label="threshold",xlabel="simulation seed",ylabel="initial loss")
############
# Extract values of the learning speed and steady state loss
############

# All values
lsAll = loop_3(postStores,getValF,[:learningSpeed,mean])
lsAllD = loop_3(postStores,getValF,[:dynamicLs,mean])
lsAllN = loop_3(postStores,getValF,[:lsN,mean])
ssAll = loop_3(postStoresSS,getValF,[:steadyStateE,mean])
if preTrain
    ssAllD = loop_3(postStoresSS,getValF,[:dynamicSS,mean])
    ssAllN = loop_3(postStoresSS,getValF,[:ssN,mean])
else
    ssAllD = loop_3(postStores,getValF,[:dynamicSS,mean])
    ssAllN = loop_3(postStores,getValF,[:ssN,mean])
end

############# Select values for a subset of learning steps gammas 
gammaI=1:1:Nmus-1 # interval of gammas to select
# gammaI=1:1:10 # interval of gammas to select
gammaIVar = [gammaI for i=1:length(NsNorm)]


############### Select subset of simulations ###############
int =1:length(postStores) # simulations to select
# int = [1,2,4,6,7,8,9,10,11,13,14,15,17,18,19,20,21,22]
# int = [6,8,9,10,11,13,14,17,21]
# int = [6,8,9,10,11,13,17,21]
# int = [22]

# arrays of learning performance variables for subset of gammas 
lsAllS = [[lsAll[i][j][gammaIVar[j]]  for j=1:length(Ns)] for i=int]
ssAllS = [[ssAllN[i][j][gammaIVar[j]]  for j=1:length(Ns)] for i=int] 
lsAllSN = map(int) do i
    if lsAll[i][1][1]!=0.0
        return [lsAll[i][j]./lsAll[i][1][1] for j=1:length(Ns)]
    else 
        return [lsAll[i][j]./lsAllD[i][1][1] for j=1:length(Ns)]
    end
end
ssAllSN = [[ssAllN[i][j]./ssAllN[i][1][1]  for j=1:length(Ns)] for i=int] 

# selected learning steps
musS = [mus[i][gammaIVar[i]] for i=1:length(Ns)] # learning steps selected


############
# Prepare labels and paths for plots
############
mkpath(plotsdir(path)) # make directory to save plots

xlbl = "expansion ratio (q)" # xlabel for plots 

# label of expansion ratio for plots
lbl = string.("q=",NsNorm)
lbl = reshape(lbl,(1,length(lbl)))

############
# optimal values lsOpt and ssOpt muOpt for each size each sim 
############

# optimal over all learning steps gammas for each sim and N
lsOpt = [[maximum(lsAllS[j][i][:]) for i=1:length(Ns)] for j=1:length(lsAllS)]
ssOpt = [[minimum(ssAllS[j][i][:]) for i=1:length(Ns)] for j=1:length(ssAllS)]
ssOpt2 = [[ssAllS[j][i][argmax(lsAllS[j][i][:])] for i=1:length(Ns)] for j=1:length(ssAllS)]
lsOptN = [[maximum(lsAllSN[j][i][:]) for i=1:length(Ns)] for j=1:length(lsAllSN)]
ssOptN = [[minimum(ssAllSN[j][i][:]) for i=1:length(Ns)] for j=1:length(ssAllSN)]
ssOpt2N = [[ssAllSN[j][i][argmax(lsAllS[j][i][:])] for i=1:length(Ns)] for j=1:length(ssAllSN)]

# colorsLines =  ["#66c2a5","#fc8d62","#8da0cb"]
colorsLines=["#a6cee3","#1f78b4","#b2df8a","#33a02c"]

# maximal learning speed over all gammas
lt = :scatter
saveLbl = plotsdir(path,"lsOpt.pdf")
llbl = [L"$\nu^*$"]
plot_mean(NsNorm,lsOpt,llbl,xlbl,L"optimal learning speed ($\nu^*$)",saveLbl,false,lt,colorsLines,7)
plot_mean(NsNorm,lsOptN,llbl,xlbl,L"optimal learning speed ($\nu^*$)",plotsdir(path,"lsOptN.pdf"),false,lt,colorsLines,7)

# minimal ss over all gammas
saveLbl = plotsdir(path,"ssOpt.pdf")
llbl = [L"$\xi^*$"]
plot_mean(NsNorm,ssOpt,llbl,xlbl,L"optimal steady state loss ($\xi^*$)",saveLbl,false,lt,colorsLines,7)

# optimal ss value for the gamma that gives optimal learning speed
saveLbl = plotsdir(path,"ss_atOptls.pdf")
llbl = [L"$\xi^*$"]
plot_mean(NsNorm,ssOpt2,llbl,xlbl,"steady state loss at\n optimal learning speed ",saveLbl,false,lt,colorsLines,7)
plot_mean(NsNorm,ssOpt2N,llbl,xlbl,"steady state loss at\n optimal learning speed ",plotsdir(path,"ss_atOptlsN.pdf"),false,lt,colorsLines,7)

## Values of learning step gamma for optimal ls and ss
muOptLs = [[musS[i][argmax(lsAllS[j][i][:])] for i=1:length(Ns)] for j=1:length(lsAllS)]
muOptss = [[musS[i][argmin(ssAllS[j][i][:])] for i=1:length(Ns)] for j=1:length(ssAllS)]

saveLbl = plotsdir(path,"gammaOptLs.pdf")
llbl = [L"$\gamma^*_{\nu}$"]
plot_mean(NsNorm,muOptLs,llbl,xlbl,L"optimal learning step ($\gamma^*_{\nu}$)",saveLbl,false,lt,false,7)

saveLbl = plotsdir(path,"gammaOptss.pdf")
llbl = [L"$\gamma^*_{\xi}$"]
plot_mean(NsNorm,muOptss,llbl,xlbl,L"optimal learning step ($\gamma^*_{\xi}$)",saveLbl,false,lt,false,7)

############
# scatter plot ls vs ss for different subsets of Ns
############
"""
    plot scatter plot of ss vs ls for a subset of sizes with indeces sizeIs
"""
function plot_scatter_Nint(ssAll,lsAll,sizeIs,slbl,stdErr=false,gammaI=false,ms=4)
    ssM = map(1:length(ssAll)) do i
        ssAll[i][sizeIs]
    end
    lsM = map(1:length(lsAll)) do i
        lsAll[i][sizeIs]
    end
    plot_scatterMean(ssM,lsM,lbl[sizeIs],gammaI,plotsdir(path,slbl),false,ms,stdErr)
end

# multiple scatter plots for different subset of Ns. If we select all, the graph is too crowded
plot_scatter_Nint(ssAllS,lsAllS,[1],"ss_vs_ls_1.pdf",true)
plot_scatter_Nint(ssAllS,lsAllS,2:5:length(Ns),"ss_vs_ls_5.pdf",true)
# plot_scatter_Nint(ssAllS,lsAllS,[1,9],"ss_vs_ls_1_9.pdf",true)
plot_scatter_Nint(ssAllS,lsAllS,[1,3,length(Ns)],"ss_vs_ls_1_3_end.pdf",true,false,6)
plot_scatter_Nint(ssAllS,lsAllS,[2,4,length(Ns)],"ss_vs_ls_2_4_end.pdf",true,false,6)
plot_scatter_Nint(ssAllS,lsAllS,[2,length(Ns)],"ss_vs_ls_2_end.pdf",true,false,6)
plot_scatter_Nint(ssAllS,lsAllS,[4],"ss_vs_ls_4.pdf",true,false,6)


#######
# Make cloud plot
#######

# # make a dataframe with ls and ss for each seed and Ns
# add learning speed normalised by the value for smallest learning step 
# do for each size and simulation differently

# dfAll = DataFrame(seed = seeds[1],Ns=Ns, ls = lsAllN[1], ss=ssAllN[1])
# for i=2:length(seeds)
#     append!(dfAll,DataFrame(seed = seeds[i],Ns=Ns, ls = lsAllN[i], ss=ssAllN[i]))
# end

""" 
    normalise ls by the ls of the smallest net smallest gamma
    normalise ss by the ss of the smallest net and smallest gamma
"""
function dfEntries(i,j,seeds,Ns,mus,lsAll,ssAllN)
    if lsAll[i][1][1]!=0.0 # normalise by the ls of the smallest net smallest gamma
        lsNorm = lsAll[i][j]./lsAll[i][1][1]
    else # if zero initial ls use dynamic ls
        lsNorm = lsAll[i][j]./lsAllD[i][1][1]
    end 
    return DataFrame(seed = seeds[i], N=Ns[j], mu=mus[j], ls = lsAll[i][j], ss=ssAllN[i][j],lsN = lsNorm, ssN = ssAllN[i][j]./ssAllN[i][1][1])
end

####
# Define the dataframe
###
dfAllE = DataFrame()
for i=1:length(seeds)
    for j=1:length(Ns)
        append!(dfAllE,dfEntries(i,j,seeds,Ns,mus,lsAll,ssAllN))
    end
end
colMap = Dict()
for i=1:ncol(dfAllE)
    a = propertynames(dfAllE)[i]
    colMap[a] = i
end

###
# plot learning speed vs gamma and ss vs gamma for different seeds
###
sizeIs = [1,9]

plotPerSize(sizeIs,Ns,dfAllE,3,4,5,lbl,plotsdir(path,"linePlots$sizeIs.pdf"))
plotPerSize(sizeIs,Ns,dfAllE,3,6,7,lbl,plotsdir(path,"linePlotsN$sizeIs.pdf"))


plotPerSeed(seeds,sizeIs,Ns,dfAllE,3,4,plotsdir(path,"ls_exp$sizeIs.pdf"),"learning step","learning speed")
plotPerSeed(seeds,sizeIs,Ns,dfAllE,3,6,plotsdir(path,"lsN_exp$sizeIs.pdf"),"learning step","normalised learning speed")
plotPerSeed(seeds,sizeIs,Ns,dfAllE,3,5,plotsdir(path,"ss_exp$sizeIs.pdf"),"learning step","steady state loss")
plotPerSeed(seeds,sizeIs,Ns,dfAllE,3,7,plotsdir(path,"ssN_exp$sizeIs.pdf"),"learning step","normalised ss")  

#########
# scatter cloud of ss vs ls for all simulations with trend line
##########
namedColors = ["Blues","Greens","Oranges","Purples","Reds","Grays"]
ms = 5

scatterCloud(sizeIs,Ns,dfAllE,colMap[:ss],colMap[:ls],plotsdir(path,"scatterCloud$sizeIs.pdf"),"steady state loss","learning speed",namedColors)
scatterCloud(sizeIs,Ns,dfAllE,colMap[:ssN],colMap[:lsN],plotsdir(path,"scatterCloudN$sizeIs.pdf"),"normalised steady state loss","normalised learning speed",namedColors)
plot!(xlims=(0,10),ylims=(0,40))
savefig(plotsdir(path,"scatterCloudN_zoom$sizeIs.pdf"))

scatterPerSeed(seeds,sizeIs,Ns,dfAllE,colMap[:ss],colMap[:ls],plotsdir(path,"scatterCloud_exp$sizeIs.pdf"),"steady state loss","learning speed",namedColors,8)
scatterPerSeed(seeds,sizeIs,Ns,dfAllE,colMap[:ssN],colMap[:lsN],plotsdir(path,"scatterCloud_expN$sizeIs.pdf"),"normalised ss","normalised ls",namedColors,8)


############
# Line plots of mean ls and ss (both method of computation) for all Ns and gammas
############
# int =1:length(seeds)
cm = palette(:heat,length(NsNorm)+1)[2:end]

xlbl=L"\gamma"
plotLines(lsAll,int,mus,lbl,xlbl,"Learning speed","ls_vs_mus.pdf",cm)
plotLines(lsAllD,int,mus,lbl,xlbl,"Learning speed","ls_vs_musD.pdf",cm)
plotLines(lsAllN,int,mus,lbl,xlbl,"Learning speed","ls_vs_musN.pdf",cm)
plotLines(ssAll,int,mus,lbl,xlbl,"Steady state loss","ss_vs_mus.pdf",cm)
plotLines(ssAllD,int,mus,lbl,xlbl,"Steady state loss","ss_vs_musD.pdf",cm)
plotLines(ssAllN,int,mus,lbl,xlbl,"Steady state loss","ss_vs_musN.pdf",cm)

############
# Plot task loss for different gammas for one size
############
# label for plots of different learning steps
lbl2 = string.(L"\gamma=",mus[1])
lbl2 = reshape(lbl2,(1,length(lbl2))) 
lblMuV = map(1:length(Ns)) do i
    string.(L"\gamma=",round.(mus[i],digits=4))
end

# Plot task loss for different gammas for one size
postStore = postStores[1][1]
Fs= map(postStore) do s
    s[:taskError]
end
plot(t_save,vcat(Fs...)',lw=3,label=lbl2)
plot!(xlabel="trajectory time")
plot!(ylabel="task loss")
savefig(plotsdir(path,"taskErrors.pdf")) 

# s =[9, 11, 12, 14, 15, 16, 18, 19, 23, 25, 26, 27, 28, 29, 30, 32, 33]