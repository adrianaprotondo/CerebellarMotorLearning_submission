# Cerebellar Online Motor Control

This code goes alongside [this manuscript](https://www.biorxiv.org/content/10.1101/2022.10.20.512268v3). It contains the julia scripts to run the simulations in the paper. 
**Disclaimer**: the code is not intended for distribution, may lack sufficient annotations or instructions. 

![motor control system](plots/Summary/system_f.jpeg)

This is a simple implementation of the motor control system above. A feedforward (cerebellar cortex) and feedback system (motor system) control a plant (musculoskeletal system). The task is for the output of the plant $y(t)$ to match a reference trajectory $r(t)$. 
The cerebellar cortex module can adapt its weights online, while controlling the plant, to improve its internal model of the plant. 

The main goal is to evaluate learning performance for different network architectures of the cerebellar cortex module. We test different learning rules for the weights of the cerebellar-like network. Most of the details of the system are given in the methods section of the manusctipt.

It is 'plug and play' in the sense that you can create, swap, and connect components easily. You can implement different forms of plants, reference trajectories, and controllers. 
It's pretty easy to create your own components by adding to the `types.jl` and  `systemComponents_functions.jl` page.

The default implementations are as follows:
  - reference trajectory is a sum of sinusoidals approximating a Orrnstein-Uhlenbeck process. 
  - motor system is a PID controller
  - cerebellar cortex is a feedforward neural network with one hidden layer with tanh activation function and a single linear output. The input weights are static and the output weights are adaptable. 
  - the musculoskeletal system is a linear plant. 

<!-- To get started we recommend going over the notebook `notebooks/LMS_test.ipynb`.  -->

<!-- ## Demo-Preview -->

<!-- ## Installation 
To use this project, first clone the repo on your device using the commands below:
 -->


## Running
The whole project is implemented in [Julia](https://docs.julialang.org/en/v1/). Julia allows for rapid calculation of gradients through the whole dynamic system. 

This is project is an 'unregistered package'. To run the project

1. Download [Julia 1.6](https://julialang.org/downloads/#long_term_support_release). Note that there might be compatibility problems to run the code in newer Julia versions. 
2. cd to the folder where you want to clone the project.
3. clone the repo on your device
```
git init
git clone ....... 
```

4. Open a Julia terminal. 
   
   1. **On VSCode** The easiest is to install the [Julia client](https://github.com/julia-vscode/julia-vscode#installing-juliavs-codevs-code-julia-extension) in [VSCode](https://code.visualstudio.com/Download).
   2. You can also just use a normal terminal. In the latter case, make sure you add and use Revise.jl before doing anything else. 
      ```
      ]
      add Revise
      ```

3. Running the code. There are multiple ways to run the code. 
Remember that in Julia, the first time you use a package it takes a really long time in Julia (compiling). Same for functions and plotting. But the second time is really quick.

### Running notebooks .ipynb
To run notebooks found in `notebooks/` folder.

#### With VSCode

1. Open VSCode
2. Add the [Jupyter extension to VSCode](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).
3.  Open the project folder `CerebellarMotorLearning`  
4.  Open one of the .ipynb in `notebooks/`
5.  Select the `Julia-1.6` kernel
6.  You can now run the cells 

#### On a terminal
1. Go to the project folder `CerebellarMotorLearning` 
2. Open a julia terminal by typing `julia` or `julia-1.6`
2. Install [IJulia](https://github.com/JuliaLang/IJulia.jl#quick-start) 
    ``` 
    ] 
    add IJulia
    ```
3. If you already have Python/Jupyter installed on your machine, you can then launch the notebook server the usual way by running `jupyter notebook` in the terminal. otherwise type the following on the Julia terminal
      ```
      using IJulia
      notebook()  
      ```

<!-- Add Pluto package
```
]
add Pluto
```

Run Pluto 
```
using Pluto 
Pluto.run()
``` -->

### Running scripts 

#### In VSCode
In VSCode you can directly open one of the scripts and run `Julia:Execute active File in new REPL`
  
#### In Julia terminal
In a julia terminal, ']' switches to the package manager, and ';' switches to the shell. If you switch to the package manager and write
```
]
activate . 
update
```
then you install and activate the environment for this package. Then type:
```
using CerebellarMotorLearning 
```
This might take a while as it is precompiling the whole package. But it will speed up running times later. 
Then switch to shell (with `;`), and `cd` to the scripts. Again, you can activate the environment in that folder, with
```
activate .
update
```

At this point you can type
`include("xxx.jl")` to run a script.
Simultaneously, (thanks to Revise.jl), you can alter functions and code in the FeedbackLearning package, and that will immediately be reflected in your code that calls it. 


## Scripts

### Helper functions
- `sizeSim_functions.jl`: functions to simulate different network sizes.
    - functions to build expanded systems with size N given original network with some parameters 
    - function build_systems_sim returns systems with different net sizes given by Ns with random seed randomSeed
    - function build_systems_sim_K:  build all the systems with sizes Ns and input sparsity Kss but same ref, plant... keep same W for different K. expand W with zeros for different N. returns vector of vector of systems with length(Kss)xlength(Ns) 
    - function build_systems_sim_K_KNconst: same as above but keeping the number of input connections to Kss[end]*N (i.e. to the largest size)
    - plotting functions
    - helper functions


## Performance tips

- Only compatible with v1.6. Earlier versions raise errors with Modelingtoolkit.jl and with jld2 loading variables.
- Important to keep Flux under v0.12.9. Newer versions raise error with Flux.destructure() and ModellingToolkit.jl

