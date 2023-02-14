#####
# Produce graphs of learning speed and steady state loss 
# for different input expansions Ns, learning steps and varBS (limit of bias distribution)
# We test two different ways of computing the learning parameters
# 1. Fit the task loss as an exponential decay c_1 e^(-c_2 t )+c_3. 
#       the learning speed is c_2 and steady state loss is c_3
# 2. Learning speed is the mean over first t_ls epochs of νₜ = -ΔFₜ/(δₜFₜ) (where F is the task loss)
#       Steady state loss is the mean of the last task loss over the last t_ls
# Use data produced by test_activationF saved in datadir(path)
# Set simModel, path and pretrain used in test_activationF 
# Plots produced:
# - line plots ls and ss as a function of gammas or Ns (different color lines)
#####

using StatsPlots
using Measures

include("default.jl")
include("sizeSim_functions.jl")

simModel = "test_actF_bias_long_2"
path = string(simModel,"_",20,"_",70)
preTrain = true

##### add simulations with tanH from another folder ##### 
# make sure same seeds
addTanh = true
simModel2 = "testVar_learningNoise_NonZeroBias_short"
path2 = string(simModel2,"_",20,"_",70)

############
# Load simulation data as a Dataframe using DrWatson
############
# dfLsSS has a columns named :postStores, mu, Ns,seed
dfLsSS = collect_results(datadir("simulations",path);verbose=true,subfolders=true) # get data in the path with simulations
sort!(dfLsSS,:seed)


# values of parameters used for simulations
t_save = dfLsSS[!,:t_save][1]

t_saveSS = t_save[1]:5:trajTimeSS

############
## Extract poststores and trainstores according to seed
############
# get postStores an array of arrays size length(seeds)xlength(Ns)xlength(mus)xnumSimV

"""
    returns postStores and postStoresSS with shape 
    length(seeds)xlength(vaS)xlength(Ns)xlength(mus)xlength(numSimV)
    var is the symbol of the variable to select different postores 
    (could be :seed, :K, :varBS) depending on the simulation
"""
function extractFromDF(dfLsSS,var,preTrain=true,numP = 1)
    sort!(dfLsSS,:seed)
    # values of parameters used for simulations
    seeds = unique(dfLsSS[!,:seed])
    mus = unique(dfLsSS[!,:mu])[1]
    Ns = unique(dfLsSS[!,:Ns])[1] # get net sizes
    t_save = dfLsSS[!,:t_save][1]
    numSimV = [unique(filter(:seed => ==(s),dfLsSS)[!,:numSimV])[1] for s in seeds]
    sort!(dfLsSS,var)
    varS = unique(dfLsSS[!,var])   
    # Nmus = length(mus[1]) # number of different mus for each size

    NsInt = 1:length(Ns)
    Ns = Ns[NsInt]
    # NsNorm = NsNorm[NsInt] # expansion ratio
    mus = mus[NsInt]

    # get all poststores has shape numSimxlength(Ns)xlength(mus)xnumSimV
    postStores = map(seeds) do s
        map(varS) do ss # get postStore for each net size for a given seed
            filter(:seed => ==(s), filter(var => ==(ss),dfLsSS))[:,:postStores][numP][NsInt]
        end
    end
    # initial loss (before learning) for each seed (equal for all sizes by design)
    initialLoss = [[postStores[j][i][1][1][1][:taskError][1,1] for j=1:length(seeds)] for i=1:length(varS)]

    if preTrain # get post-stores to compute SS loss if there was pretrain    
        t_saveSS = t_save[1]:5:trajTimeSS
        postStoresSS = map(seeds) do s
            map(varS) do ss # get postStore for each net size for a given seed
                filter(:seed => ==(s), filter(var => ==(ss),dfLsSS))[:,:postStoresSS][numP][NsInt]
            end
        end

        loop_5(postStoresSS,computeDynamicSS!,[50,false]) # compute dynamic ss
        # compute ss from normalised task loss
        map(1:length(postStoresSS)) do i
            map(1:length(postStoresSS[i])) do j
                loop_3(postStoresSS[i][j],computeNormalisedTaskE!,[initialLoss[j][i]])
            end
        end
        # ss is mean of task loss over the second half of traj
        loop_5(postStoresSS,computeDynamicSS!,[Int(length(t_saveSS)/2),false,:taskErrorN,:ssN])
    end
    # Compute learning speed and steady state loss from mean (described above)
    loop_5(postStores,computeDynamicLs!,[t_save,100]) # compute dynamic ls 
    loop_5(postStores,computeDynamicSS!,[50]) # compute dynamic ss
    # compute normalised task error and learning performance with respect to the normalised value
    # loop_4(postStores,computeNormalisedTaskE!,[]) # compute taskErrorN 
    # loop_4(postStores,computeLsSSFit!,[t_save,:taskErrorN]) # compute dynamic ls 
    return postStores, postStoresSS, varS, Ns, mus, seeds, numSimV
end

#######
# Make dataframe learning speed and steady state loss
#######
"""
    return a dataframe with columns given by varK, keyP,keyPSS, keyP_norm, keyPSS_norm 
    indeces is an array of length 5 giving the indeces for each one of the variables in varA with corresponding name varK 
    indeces, varA and varK must have the same length = 5
    keyP is the key to extract from p and save in dataframe 
    keyPSS is the key to extract from pSS and save in Dataframe
    keyM is the key of the variable to take mean and save in dataframe 
    p and pSS are postStores with 5 layers of arrays with each level having the same length as varA arrays 
    for example have shape length(seeds)xlength(varBS)xlength(Ns)xlength(mus[1])xlength(numSimV)
    add as many rows as there are different mus
    are normalised by the value for ls for the smallest Ns, smallest mu (i.e. third and fourth level) for each seed, varBS, simV (first and second level)
"""
function dfEntries_5(indeces,varA,keyP,keyPSS,p,pSS,varK=[:seed,:varBS,:N,:mu,:simV],keyM=false,keyMDefault=1.0)
    i,j,k,kk,l=indeces
    lsVal = [p[i][j][k][m][l][keyP] for m=1:length(p[i][j][k])]
    ssVal = [pSS[i][j][k][m][l][keyPSS] for m=1:length(pSS[i][j][k])]
    if p[i][1][1][1][l][keyP]!=0.0 && !isnan(p[i][1][1][1][l][keyP]) # normalise by smallest N and smallest mu
        println(p[i][1][1][1][l][keyP])
        lsNormP = [p[i][j][k][m][l][keyP]./p[i][1][1][1][l][keyP] for m=1:length(p[i][j][k])]
    else # if zero initial ls use try with second value
        println(p[i][2][1][1][l][keyP])
        lsNormP = [p[i][j][k][m][l][keyP]./p[i][1][1][2][l][keyP] for m=1:length(p[i][j][k])]
    end 
    if pSS[i][1][1][1][l][keyPSS]!=0.0 # normalise by smallest N and smallest mu
        lsNormPSS = [pSS[i][j][k][m][l][keyPSS]./pSS[i][1][1][1][l][keyPSS] for m=1:length(pSS[i][j][k])]
    else # if zero initial ls use try with second value
        lsNormPSS = [pSS[i][j][k][m][l][keyPSS]./p[i][1][1][2][l][keyPSS] for m=1:length(pSS[i][j][k])]
    end 
    if keyM !=false # if save mean of a variable 
        if keyMDefault == false # use values from the postStore
            d=DataFrame(vcat([varK[m] => varA[m][indeces[m]] for m=1:lastindex(varK)], keyP => lsVal, (Symbol(keyP,"_norm") => lsNormP), keyPSS => ssVal, Symbol(keyPSS,"_norm") => lsNormPSS, keyM => [mean(p[i][j][k][m][l][keyM]) for m=1:length(p[i][j][k])]))
        else # use default value 
            d=DataFrame(vcat([varK[m] => varA[m][indeces[m]] for m=1:lastindex(varK)], keyP => lsVal, (Symbol(keyP,"_norm") => lsNormP), keyPSS => ssVal, Symbol(keyPSS,"_norm") => lsNormPSS, keyM => keyMDefault*ones(length(p[i][j][k])) ))
        end
    else
        d=DataFrame(vcat([varK[m] => varA[m][indeces[m]] for m=1:lastindex(varK)], keyP => lsVal, (Symbol(keyP,"_norm") => lsNormP), keyPSS => ssVal, Symbol(keyPSS,"_norm") => lsNormPSS))
    end
    return d
end

####
# Define the dataframe
###
varSym = :varBS
filter!(:postStores => x -> !(ismissing(x) || isnothing(x)),dfLsSS)
sort!(dfLsSS,:seed)
postStores, postStoresSS, varS, Ns, mus, seeds, numSimV = extractFromDF(dfLsSS,varSym,true);

NsNorm = Ns./num_nn_inputs

varA = [seeds,varS,Ns,mus,1:numSimV[1]]
varKeys=[:seed,:varBS,:N,:mu,:simV]
lsSym = :learningSpeed # value to extract as learning speed
# lsSym = :dynamicLs
ssSym = :ssN # value to extract as ss loss
# ssSym = :steadyStateE
dfAllE = DataFrame()
for i=1:length(seeds)
    for j=1:length(varS)
        for k=1:length(Ns)
            for l=1:numSimV[i]
                varA[end] = 1:numSimV[i]
                append!(dfAllE,dfEntries_5([i,j,k,k,l],varA,lsSym,ssSym,postStores,postStoresSS,varKeys,:codingLevel,false))
            end
        end
    end
end

if addTanh #add tanh simulations
    tanBS = 10.0
    dfLsSS2 = collect_results(datadir("simulations",path2);verbose=true) # get data in the path with simulations
    postStores2, postStoresSS2, varS2, Ns2, mus2, seeds2, numSimV2 = extractFromDF(dfLsSS2,varSym,true);
    varA2 = [seeds2,[tanBS],Ns2,mus2,1:numSimV2[1]]

    for i=1:length(seeds2)
        for j=1:length(varS2)
            for k=1:length(Ns2)
                for l=1:numSimV2[i]
                    varA2[end] = 1:numSimV2[i]
                    append!(dfAllE,dfEntries_5([i,j,k,k,l],varA2,lsSym,ssSym,postStores2,postStoresSS2,varKeys,:codingLevel))
                end
            end
        end
    end
    varBSs = vcat(varS,tanBS)
else
    varBSs = varS
end

### filter out some anomalies
filter!(lsSym => x -> !(ismissing(x) || isnothing(x) || isnan(x)),dfAllE)
filter!(lsSym => x -> x<1.0,dfAllE)
filter!(Symbol(lsSym,"_norm") => x -> !(ismissing(x) || isnothing(x) || isnan(x)),dfAllE)

gdf = groupby(dfAllE,[:seed,varSym,:N,:mu])
subProp = propertynames(dfAllE)[6:end] # prop to take mean and 
dfAllMean = combine(gdf,vcat([prop=>mean=>Symbol(prop,"_mean") for prop in subProp],[prop=>std=>Symbol(prop,"_std") for prop in subProp]))
colMap = Dict()
for i=1:ncol(dfAllMean)
    a = propertynames(dfAllMean)[i]
    colMap[a] = i
end

############
# Prepare labels and paths for plots
############
mkpath(plotsdir(path)) # make directory to save plots

xlbl = "expansion ratio (q)" # xlabel for plots 

# label of expansion ratio for plots
lbl = string.("q=",NsNorm)
lbl = reshape(lbl,(1,length(lbl)))
lblB = string.(L"b_m=",varBSs)
lblB = reshape(lblB,(1,length(lblB)))
###
# plot learning speed vs gamma and ss vs gamma for different varBS
###
# sizeIs = [2,9]
sizeIs = [1,2]
# sizeIs = collect(3:2:9)

lsMS = Symbol(lsSym,"_mean")
ssMS = Symbol(ssSym,"_mean")
lsStdS = Symbol(lsSym,"_std")
ssStdS = Symbol(ssSym,"_std")

sort!(dfAllMean,:mu)

# sizeIs = [1,2,3]

s = seeds[end]
df = sort!(filter(:seed => ==(s),dfAllMean),:mu)
# df = filter(row -> (row.seed ==(9)|| row.varBS==(1.0)),dfAllMean)
plotPerSize(sizeIs,Ns,df,colMap[:mu],colMap[lsMS],colMap[ssMS],lbl,plotsdir(path,"linePlotMs_log_$sizeIs _seed-$s.pdf"),colMap[lsStdS],colMap[ssStdS];groupBy=colMap[:varBS],axScale=:log10)
plotPerSize(sizeIs,Ns,df,colMap[:mu],6,8,lbl,plotsdir(path,"linePlotMs_norm_log_$sizeIs _seed-$s.pdf"),11,13;groupBy=colMap[:varBS],axScale=:log10)
# plotPerSize(sizeIs,Ns,dfAllMean,colMap[:mu],colMap[:lsN_mean],colMap[:ssN_mean],lbl,plotsdir(path,"linePlotsN$sizeIs.pdf"),colMap[:lsN_std],colMap[:ssN_std])

######## ls and ss vs coding level one mu two Ns ##########
# colors = ["#1f78b4" "#33a02c"]
colors = ["#65B9E7" "#76C76A"]

s = seeds[end]
musS = [mus[i][2] for i=1:length(mus)]
dfi = sort(filter(:mu => in(musS),filter(:seed => ==(s),dfAllMean)),:varBS)
@df dfi scatter(:codingLevel_mean,:learningSpeed_mean,xlabel="coding level", ylabel="learning speed",colour = colors, group=:N,yerr=:learningSpeed_std,ms=10)
savefig(plotsdir(path,"LSvsCL_seed_$s mus-$musS.pdf"))

# @df dfi scatter(:codingLevel_mean,:steadyStateE_mean, xlabel="coding level", ylabel="steady state loss", colour = colors, group=:N,yerr=:steadyStateE_std)
@df dfi scatter(:codingLevel_mean,:ssN_mean, xlabel="coding level", ylabel="steady state loss", colour = colors, group=:N,yerr=:ssN_std,ms=10)
savefig(plotsdir(path,"ssvsCL_seed_$s mus-$musS.pdf"))

# s = seeds[2]
N = Ns[end]
musS = mus[end][2]
dfi = sort(filter(:N => ==(N),filter(:mu => in(musS),filter(:seed => ==(s),dfAllMean))),:varBS)
@df dfi scatter(:codingLevel_mean,:learningSpeed_mean,xlabel="coding level", ylabel="learning speed",colour = colors, group=:N,yerr=:learningSpeed_std,ms=10)
savefig(plotsdir(path,"LSvsCL_seed_$s N-$N mus-$musS.pdf"))

# @df dfi scatter(:codingLevel_mean,:steadyStateE_mean, xlabel="coding level", ylabel="steady state loss", colour = colors, group=:N,yerr=:steadyStateE_std)
@df dfi scatter(:codingLevel_mean,:ssN_mean, xlabel="coding level", ylabel="steady state loss", colour = colors, group=:N,yerr=:ssN_std,ms=10)
savefig(plotsdir(path,"ssvsCL_seed_$s N-$N mus-$musS.pdf"))

# s = seeds[2]
# N = Ns[end]
musS = mus[end]
dfi = sort(filter(:N => ==(N),filter(:mu => in(musS),filter(:seed => ==(s),dfAllMean))),:varBS)
@df dfi scatter(:codingLevel_mean,:learningSpeed_mean,xlabel="coding level", ylabel="learning speed",colour = colors, group=:mu,yerr=:learningSpeed_std,ms=10)
savefig(plotsdir(path,"LSvsCL_seed_$s N-$N mus-$musS.pdf"))

@df dfi scatter(:codingLevel_mean,:ssN_mean, xlabel="coding level", ylabel="steady state loss", colour = colors, group=:mu,yerr=:ssN_std,ms=10)
savefig(plotsdir(path,"ssvsCL_seed_$s N-$N mus-$musS.pdf"))


@df dfAllMean scatter(:N,:codingLevel_mean,group=:varBS)
plot!(xlabel="number of GC",ylabel="mean coding level")
savefig(plotsdir(path,"codingLevelvsGC.pdf"))

dfi = filter(:varBS => in(varBSs[1:end-1]),dfAllMean)
@df dfi scatter(:varBS,:codingLevel_mean,group=:N)
plot!(xlabel="b",ylabel="mean coding level")
savefig(plotsdir(path,"codingLevelvsVarBS.pdf"))

########## mean over all seeds ###############
gdf = groupby(filter(:seed => in([9,12]),dfAllE),[varSym,:N,:mu])
subProp = propertynames(dfAllE)[6:end] # prop to take mean and 
dfAllMean2 = combine(gdf,vcat([prop=>mean=>Symbol(prop,"_mean") for prop in subProp],[prop=>std=>Symbol(prop,"_std") for prop in subProp]))


musS = [mus[i][2] for i=1:length(mus)]
dfi = sort(filter(:mu => in(musS),dfAllMean2),:varBS)
@df dfi scatter(:codingLevel_mean,cols(4),xlabel="coding level", ylabel="learning speed",colour = colors, group=:N,yerr=cols(9))
savefig(plotsdir(path,"LSvsCL_mus-$musS.pdf"))

@df dfi scatter(:codingLevel_mean,cols(5),xlabel="coding level", ylabel="learning speed",colour = colors, group=:N,yerr=cols(10))
savefig(plotsdir(path,"LSvsCL_norm_mus-$musS.pdf"))

@df dfi scatter(:codingLevel_mean,:ssN_mean, xlabel="coding level", ylabel="steady state loss", colour = colors, group=:N,yerr=:ssN_std)
savefig(plotsdir(path,"ssNvsCL_mus-$musS.pdf"))
# @df dfi scatter(:codingLevel_mean,:steadyStateE_mean, xlabel="coding level", ylabel="steady state loss", colour = colors, group=:N,yerr=:steadyStateE_std)
@df dfi scatter(:codingLevel_mean,:ssN_norm_mean, xlabel="coding level", ylabel="steady state loss", colour = colors, group=:N,yerr=:ssN_norm_std)
savefig(plotsdir(path,"ssNvsCL_norm_mus-$musS.pdf"))


N = Ns[end]
musS = mus[end][2]
dfi = sort(filter(:N => ==(N),filter(:mu => in(musS),dfAllMean2)),:varBS)
@df dfi scatter(:codingLevel_mean,cols(4),xlabel="coding level", ylabel="learning speed",colour = colors, group=:N,yerr=cols(9))
savefig(plotsdir(path,"LSvsCL_N-$N mus-$musS.pdf"))
@df dfi scatter(:codingLevel_mean,cols(5),xlabel="coding level", ylabel="learning speed",colour = colors, group=:N,yerr=cols(10))
savefig(plotsdir(path,"LSvsCL_norm_N-$N mus-$musS.pdf"))

@df dfi scatter(:codingLevel_mean,:ssN_mean, xlabel="coding level", ylabel="steady state loss", colour = colors, group=:N,yerr=:ssN_std)
savefig(plotsdir(path,"ssNvsCL_N-$N mus-$musS.pdf"))
@df dfi scatter(:codingLevel_mean,:ssN_norm_mean, xlabel="coding level", ylabel="steady state loss", colour = colors, group=:N,yerr=:ssN_norm_std)
savefig(plotsdir(path,"ssNvsCL_norm_N-$N mus-$musS.pdf"))



N = Ns[end]
musS = mus[end][1]
dfi = sort(filter(:N => ==(N),filter(:mu => in(musS),dfAllE)),:varBS)
@df dfi scatter(:codingLevel,cols(4),xlabel="coding level", ylabel="learning speed",colour = colors, group=:N)
savefig(plotsdir(path,"LSvsCL_all_N-$N mus-$musS.pdf"))

@df dfi scatter(:codingLevel,:ssN_norm, xlabel="coding level", ylabel="steady state loss", group=:seed)
savefig(plotsdir(path,"ssvsCL_N-$N mus-$musS.pdf"))



##########  learning speed and ss vs coding level###########

int = 1:length(varBSs)
int = 1:6
N = Ns[end]
s = seeds[1]
dfi = filter(:seed => ==(s),filter(:N => ==(N),dfAllMean))
scatterCloud(int,varBSs,dfi,colMap[:codingLevel_mean],colMap[lsMS],lblB,plotsdir(path,"ls_codingLevel_seed-$s _Ns-$N.pdf"),"coding level","learning speed",false,colMap[:codingLevel_std],colMap[lsStdS];filterBy=:varBS,trend=false)
### steady state 
scatterCloud(int,varBSs,dfi,colMap[:codingLevel_mean],colMap[ssMS],lblB,plotsdir(path,"ss_codingLevel_seed-$s _Ns-$N.pdf"),"coding level","steady state loss",false,colMap[:codingLevel_std],colMap[ssStdS];filterBy=:varBS,trend=false)

### log 
scatterCloud(int,varBSs,dfi,colMap[:codingLevel_mean],colMap[lsMS],lblB,plotsdir(path,"ls_codingLevel_log_seed-$s _Ns-$N.pdf"),"coding level","learning speed",false,colMap[:codingLevel_std],colMap[lsStdS];filterBy=:varBS,trend=false,yaxScale=:log10)
scatterCloud(int,varBSs,dfi,colMap[:codingLevel_mean],colMap[ssMS],lblB,plotsdir(path,"ss_codingLevel_log_seed-$s _Ns-$N.pdf"),"coding level","steady state loss",false,colMap[:codingLevel_std],colMap[ssStdS];filterBy=:varBS,trend=false,yaxScale=:log10)


### same figure all ls vs all ss 
s = seeds[1]
N = Ns[end]
dfi = sort(filter(:seed => ==(s),filter(:N => ==(N),dfAllMean)),:varBS)
@df dfi scatter(:codingLevel_mean,:learningSpeed_mean,xlabel="coding level", ylabel="learning speed",legend=false,yaxis=:log10)
@df dfi scatter!(twinx(),:codingLevel_mean,:ssN_mean,axis=:right,ylabel="steady state loss",color=:orange,legend=false,yaxis=:log10)
plot!(right_margin=20mm,left_margin=10mm)
savefig(plotsdir(path,"LPvsCL_all_log_seed_$s Ns-$N.pdf"))

#################
# tradeoff for each varBS
##################
namedColors = ["Blues","Greens","Oranges","Purples","Reds","Grays"]

sizeIs = [1,2,3]
map(seeds) do s
    map(varBSs) do varBS
        dfi = filter(lsSym => x -> !(ismissing(x) || isnothing(x) || isnan(x)),filter(:seed => ==(s), filter(:varBS => ==(varBS),dfAllE)))
        scatterCloud(sizeIs,Ns,dfi,8,6,lbl,plotsdir(path,"scatterCloud_all_$sizeIs _b-m_$varBS _seed_$s.pdf"),"steady state loss","learning speed",namedColors)
    end
end

##########
# Max learning speed
###########
gdf = groupby(dfAllMean,[:varBS,:N,:seed])
# subProp = propertynames(dfAllE)[4:end] # prop to take mean and 
# dfAllMax = combine(gdf,vcat([prop=>mean=>Symbol(prop,"_mean") for prop in subProp],[prop=>std=>Symbol(prop,"_std") for prop in subProp]))
dfMax = combine(gdf) do sdf
    DataFrame(first(sort(sdf, lsMS),1))
    # DataFrame(survived = sum(sdf.survived))
end


##### max ls vs coding level per seed ####
s = 11
i = 3
N = Ns[i]
bs = [-0.1,0.1,1.0]
# dfi = sort(filter(:seed => ==(s),filter(:N => ==(N),dfMax)),:varBS)
dfi = sort(filter(:varBS => x->x in bs, filter(:seed => ==(s),filter(:N => ==(N),dfMax))),:varBS)
@df dfi scatter(:codingLevel_mean,:learningSpeed_mean,yaxis=:log10,yerr=:learningSpeed_std,xlabel="coding level", ylabel="max learning speed",label=lbl[i])
i2 = 2
N = Ns[i2]
dfi = sort(filter(:varBS => x->x in bs, filter(:seed => ==(s),filter(:N => ==(N),dfMax))),:varBS)
# dfi = sort(filter(:seed => ==(s),filter(:N => ==(N),dfMax)),:varBS)
@df dfi scatter!(:codingLevel_mean,:learningSpeed_mean,yerr=:learningSpeed_std,label=lbl[i2])
savefig(plotsdir(path,"ls_codingL_log_seed_$s _varBS-$bs.pdf"))

##### ss vs coding level per seed ####
N = Ns[i]
# dfi = sort(filter(:seed => ==(s),filter(:N => ==(N),dfMax)),:varBS)
dfi = sort(filter(:varBS => x->x in bs, filter(:seed => ==(s),filter(:N => ==(N),dfMax))),:varBS)
@df dfi scatter(:codingLevel_mean,:ssN_mean,yaxis=:log10,yerr=:ssN_std,xlabel="coding level", ylabel="steady state loss at max learning speed",label=lbl[i])
N = Ns[i2]
dfi = sort(filter(:varBS => x->x in bs, filter(:seed => ==(s),filter(:N => ==(N),dfMax))),:varBS)
# dfi = sort(filter(:seed => ==(s),filter(:N => ==(N),dfMax)),:varBS)
@df dfi scatter!(:codingLevel_mean,:ssN_mean,yaxis=:log10,yerr=:ssN_std,xlabel="coding level", ylabel="steady state loss at max learning speed",label=lbl[i2])
savefig(plotsdir(path,"ss_codingL_log_seed_$s _varBS-$bs.pdf"))


# @df dfi scatter!(twinx(),:codingLevel_mean,:ssN_mean,yaxis=:log10,axis=:right,ylabel="steady state loss",color=:orange,legend=false)
# # @df dfi scatter!(twinx(),:codingLevel_mean,:ssN_mean,yaxis=:log10,axis=:right,yerr=:ssN_std,ylabel="steady state loss",color=:orange,legend=false)
# plot!(right_margin=20mm,left_margin=10mm)
# savefig(plotsdir(path,"cLvsLP_log_seed_$s Ns-$N.pdf"))

# varBS = varBSs[1]
s = 11
# dfi = sort(filter(:seed => ==(s),filter(:N => ==(N),dfMax)),:varBS)
dfi = sort(filter(:seed => ==(s),dfMax),:varBS)
# dfi = filter(:varBS => in([-0.1,0.1,1.0]),dfMax)
@df dfi scatter(:N,:learningSpeed_mean,group=:varBS,yerr=:learningSpeed_std,xlabel=xlbl,ylabel="max learning speed")
savefig(plotsdir(path,"maxLS_q_seed$s.pdf"))

@df dfi scatter(:N,:ssN_mean,group=:varBS,yerr=:ssN_std,xlabel=xlbl,ylabel="steady state loss at max learning speed")
savefig(plotsdir(path,"SSatMaxLS_q_seeds$s.pdf"))



######### plot per seed #####
plotPerSeed(varBSs,sizeIs,Ns,dfAllMean,colMap[:mu],colMap[lsMS],lbl,plotsdir(path,"ls_exp$sizeIs.pdf"),"learning step","learning speed",colMap[lsStdS];sSymb=:varBS,tLbl="min bias")
# plotPerSeed(varBSs,sizeIs,Ns,dfAllMean,colMap[:mu],colMap[:lsN_mean],lbl,plotsdir(path,"lsN_exp$sizeIs.pdf"),"learning step","normalised learning speed",colMap[:lsN_std];sSymb=:varBS,tLbl="min bias")
plotPerSeed(varBSs,sizeIs,Ns,dfAllMean,colMap[:mu],colMap[ssMS],lbl,plotsdir(path,"ss_exp$sizeIs.pdf"),"learning step","steady state loss",colMap[ssStdS];sSymb=:varBS,tLbl="min bias")
# plotPerSeed(varBSs,sizeIs,Ns,dfAllMean,colMap[:mu],colMap[:ssN_mean],lbl,plotsdir(path,"ssN_exp$sizeIs.pdf"),"learning step","normalised ss",colMap[:ssN_std];sSymb=:varBS,tLbl="min bias")  
# plotPerSeed(varBSs,sizeIs,Ns,dfAllMean,colMap[:mu],colMap[:codingLevel_mean],lbl,plotsdir(path,"codingLevel$sizeIs.pdf"),"learning step","coding level",colMap[:codingLevel_std];sSymb=:varBS,tLbl="min bias")




# """
#     return dataframe with columns
#     varKeys, pSKeys and pSSKeys
#     indeces gives index to select from varA 
#     varKeys gives the keys in dfLsSS for varA 
#         varA, indeces and varKeys should have the same length =5
#     compute extra variables in the postores
#     select variables with keys pSKeys  from the poststores 
# """
# function extractPS(indeces,varA,dfLsSS,preTrain,varKeys=[:seed,:varBS,:Ns,:mu,:simV],pSKeys=[:ls,:dynamicLs,:codingLevel],pSSKeys=[:ssN,:dynamicSS,:steadyStateE])
#     # get postores for all mus for one size and one simV
#     p = filter(varKeys[1] => ==(varA[1][indeces[1]]), filter(varKeys[2] => ==(varA[2][indeces[2]]),dfLsSS))[:,:postStores][1][indeces[3]]
#     initialLoss = p[1][1][:taskError][1,1]
#     loop_2(p,computeDynamicLs!,[t_save,100]) # compute dynamic ls 
#     loop_2(p,computeDynamicSS!,[50]) # compute dynamic ss
#     # compute normalised task error and learning performance with respect to the normalised value
#     loop_2(p,computeNormalisedTaskE!,[initialLoss]) # compute taskErrorN 
#     # loop_2(p,computeLsSSFit!,[t_save,:taskErrorN]) # compute dynamic ls 
#     pS = [p[i][indeces[5]] for i=1:length(varA[4][indeces[4]])]
#     if preTrain
#         p2 = p = filter(varKeys[1] => ==(varA[1][indeces[1]]), filter(varKeys[2] => ==(varA[2][indeces[2]]),dfLsSS))[:,:postStoresSS][1][indeces[3]] 
#         loop_2(p2,computeDynamicSS!,[50,false]) 
#         loop_2(p2,computeNormalisedTaskE!,[initialLoss])
#         loop_2(p2,computeDynamicSS!,[Int(length(t_saveSS)/2),false,:taskErrorN,:ssN])
#         pSS = [p2[i][indeces[5]] for i=1:length(varA[4][indeces[4]])]
#         pSSDF = [pSSKeys[i] => [pSS[j][pSSKeys[i]] for j=1:length(pSS)] for i=1:lastindex(pSSKeys)]
#         pSDF = [pSKeys[i] => [pS[j][pSKeys[i]] for j=1:length(pS)] for i=1:lastindex(pSKeys)]
#         d = DataFrame(vcat([varKeys[i] => varA[i][indeces[i]] for i=1:lastindex(indeces)], pSDF, pSSDF))
#     else
#         pSDF = [pSKeys[i] => [pS[j][pSKeys[i]] for j=1:length(pS)] for i=1:lastindex(vcat(pSKeys,pSSKeys))]
#         d = DataFrame(vcat([varKeys[i] => varA[i][indeces[i]] for i=1:lastindex(indeces)], pSDF))
#     end
#     return d
# end

# varA = [seeds,varBSs,Ns,mus,1:numSimV]
# varKeys=[:seed,:varBS,:Ns,:mu,:simV]
# pSKeys=[:learningSpeed,:dynamicLs,:codingLevel]
# pSSKeys=[:ssN,:dynamicSS,:steadyStateE]

# psDF = DataFrame()
# for i=1:length(seeds)
#     for j=1:length(varBSs)
#         for k=1:length(Ns)
#             for l=1:numSimV
#                 # println(i,j,k,l)
#                 append!(psDF,extractPS([i,j,k,k,l],varA,dfLsSS,preTrain,varKeys,pSKeys,pSSKeys))
#             end
#         end
#     end
# end

# if addTanh #add tanh simulations
#     dfLsSS2 = collect_results(datadir("simulations",path2);verbose=true) # get data in the path with simulations
#     seeds2 = unique(dfLsSS2[!,:seed])
#     mus2 = unique(dfLsSS2[!,:mu])[1]
#     t_save2 = dfLsSS2[!,:t_save][1]
#     numSimV2 = unique(dfLsSS2[!,:numSimV])[1]
#     varBSs2 = unique(dfLsSS2[!,:varBS]) 

#     varA2 = [seeds2,varBSs2,Ns,mus2,1:numSimV2]
#     varKeys2=[:seed,:varBS,:Ns,:mu,:simV]
#     psDF2 = DataFrame()
#     for i=1:length(seeds2)
#         for j=1:length(varBSs2)
#             for k=1:length(Ns)
#                 for l=1:numSimV2
#                     # println(i,j,k,l)
#                     append!(psDF2,extractPS([i,j,k,k,l],varA2,dfLsSS2,preTrain,varKeys2,[:learningSpeed,:dynamicLs],pSSKeys))
#                 end
#             end
#         end
#     end
# end

# # add mean coding level
# transform(psDF, :codingLevel => ByRow(mean) => :codingLevelM)
# gdf = groupby(filter(:learningSpeed => x-> !(ismissing(x) || isnothing(x) || isnan(x)), psDF),[:varBS,:seed,:simV])
