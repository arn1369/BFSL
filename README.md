# Hierarchical Sheaf Networks & Topological HMM

Implementing Sheaf Neural Networks with HMM to model market as a dynamic topological structure. We use structural coherence (Cohomology $H¹$) as alpha signal.

1. Why cohomology ?
    We don't use homology, because there isn't a cup product -> better structure.
    So why $H¹$ instead of other cohomology space $H^k$ ?
    $H⁰$ is the "Beta". When everything goes up, $H⁰$ is high. Bad for alpha.
    $H¹$ is the obstruction to global coherence. Locally, the cicle runs, but globally, we get an impossible cicle -> anomaly (structural "stress")
    $H²$ is superior order incoherence. Hard to interpret, and it takes a lot of computational workload ($O(N³)$) !

    So we have our $H¹$, the cohomology score, an indicator of structural (systemic) stress.
    When :
        1) $H¹\approx 0$ -> Coherent market (cycles are stables)
        2) $H¹ >> 0$ -> Incoherent market (cycles are distorted)

    So let's use $H¹$ instead of volatility to find the right regime !
    Well... Not exactly. H¹ alone has problems : in a global crisis, the market is structured. So it falls with it. To solve this problem, we add volatility : We use a composite signal (H1 modulated by volatility) to detect regimes. High volatility increase structural stress, so the model can distinguish 'calm coherence' (Bull) and 'crash coherence' (Bear)."

2. Why HMM ?
    Unlike usual methods, we don't use Hidden Markov Models to predict future prices : we use it to analyse the structural quality with $H¹$. The regimes are the hidden states. Here, there is 4 regimes : Bull ($H¹$ low), Normal ($H¹$ medium), Volatil ($H¹$ high), Crisis ($H¹$ very high). Be aware that these states are not particularily the market states : it only indicates the coherence of the market. So Bull state is a high-correlated market. Unlike standard HMMs, we add a Trend Filter (Moving Average) and Realized Volatility. This avoids the 'Structural Trap' described earlier where the model falls with a (coherent) structural crash.

    Also, in HMM, we implemented inertia to avoid that the model changes too fast of opinion (avoid whipsaw). The HMM is also useful to change some investment parameters like leverage, long/short, etc.

3. Why Functorial Sheaf Learning ?
    Unlike other models, Sheaf Neural Netwok uses linear applications to explain different states. This is used to model more complex relations, and to explain them ! We can invert, rotate, unphase, linear applications ! It gives the keys to translate APPL to MFST (sort of), instead of only showing the correlation (like GNN or smth else).

    The functorial part is useful to keep the topological structure between layers. In classical neural networks, geometrical structure is sometimes lost between the layers. So the functorial part is useful to keep the structural interpretability !

## Architecture

### Hierarchical Forecasting

Deep Neural Network to learn topology of the market

1) Dynamic Sheaf Laplacian : Learns the relations between assets that changes in time via Attention mechanism (to detail more). Technically, we use a stochastic (random-walk) attention matrix (numerical stability and scale-invariant diffusion). It's like a diffusion Laplacian $L = D - A$, but it's more like a diffusion operator $W = D^{-1}\cdot A$
2) Multi-scale U-Net : Micro/Macro - local volatility and global tendancy using Restrictions and Extensions (to detail more)
3) Topology Consistency ($H¹$) : Measure at which point the assets movements are "coherents" with the learned structure. High $H¹$ indicates an anomaly in the structure.

### Training

Try to separate signal and noise using

Structural Triplet Loss : Learn what is anchor (healthy market) or positive (noisy market) and negative (total chaos lol)

Try to optimize directional prediction (Sign Loss), ranking and signal reconstruction.

### HMM and Adapatative parameters

1) Regime detection : using HMM to predict regime (Normal, Bull, Volatil, Crisis) on a vol-adjusted topological signal $H¹\times (1+\sigma)$. This ensures that if the market structure is coherent (high H¹), a spike in vol will force HMM into defensive state.
2) Trading rules change with time : leverage, long/short, etc. change with the regime.

### Transformers and LSTM comparison

The nice thing about FSL is that the model "knows" if he's right or not : we have structural confidence from the model with $H¹$ (structural problem or not !). It knows when not to invest (and it's sure -or not- about it!). But we need to see the limits of this : after a certain threshold of noise, the model doesn't know when he's sure or not. Moreover, with bad learning, he's sure but about bad things ! So these steps are highly important.

Here, using LSTM is bad because it is difficult to find links between assets, and if the market has a regime change, it stinks. Of course, there exists some adjustments but fundamentally, there is a problem.

Using Transformers here would be hard too, because the transformer would see correlation everywere, and will hallucinate relations between assets. Moreover, it needs tremandus amount of data to learn !

Here FSL is native multi-assets : the model *is* the graph of relations. Moreover, Transformers and LSTM are black boxes. FSL can be interpreted, we can analyse it's structure and see what goes right/wrong in the market. Additionnaly, with Sheaf Learning, we prevent the model to learn from bad data. But the thing is that we have a strong assumption : the markets tends to an equilibrium (stable).

## Additional Notes

We can note that FSL has similar performance when we vary the seed of randomness. Nice !

## To go further

The goal is to apply FSL in more domains, notably in medical predictions data.

### Problems and possible solutions

1) Hard-Coded Gating -> use soft gating or continuous transfer function?
2) Lot of parameters in DynamicSheafLaplacian (ModuleDict for each pair) -> distribute weights ? or factorization ?
3) Adaptive Parameters using percentiles instead of values
4) Non-stationarity of $H¹$ : Fix means in HMM to define regimes. Need to update that slowly (So that if the market structurally change, the model learns it). For now, we define "Crisis" to be $H¹$ Z-score 3.0. If the mean noise of $H¹$ change, this threshold is not good anymore.
5) No fees of trading...
6) Survivor bias (we have the S&P500 (16 tickers) on assets that performed really well these last years)
7) Hard scalability on large number of assets (N² restriction matrices). Complexity of $O(N²)$. Here we have 256 matrices (16²). S&P500 : ~250,000 !! -> other architecture, like highly hierarchical (where it doesn't listen to some assets when $H¹$ is low in the domain, and when $H¹$ increase in that domain, )
8) Perpetual fight against trivial collapse, with spectral regularization. For now I don't have this problem, but maybe need to optimize it.
9) Cost of short are not updated in crisis. !! High assumption !!
10) Bad in fast crash (lag of HMM)
11) In M&A, bad ! -> MFST acquires Activision. Activision will be uncorrelated from the market to align with MSFT. $H¹$ is high and detect risk and shout the crisis.
12) Complexity ($O(N²*D²)$ with N number of asset pairs and D feature dimension). Really bad for scaling... Transformer : $O(N²\cdot D)$. A possible solution (need to confirm it with results) would be to restraint assets to "talk" with only it's k neighbors : $O(N\cdot k\cdot D²)$, more viable.
13) Better README file (finalize using latex for formulas etc) and better in-detail explanations

### Improvements

1) Learnable Residual Weights
2) Diagonal in adjacency matrix is $-10^9$ (temporary solution with the Diffusion Matrix, try to test Normalized Laplacian ?)
3) Maybe interesting to see $H²$, but at the price of interpretability -> dimensionality reduction ? what performance with $H²$ ? Are the links between assets sooo complex in real life ? Also complexity ($O(N^3)$) -> Using hypergraph to capture collective (sector) behavior -> Sheaf Hypergraph Networks (lighter than $H²$ computation)
4) To solve complexity (another solution) :
    Instead of learning a dense matrix $W_{ij}\in \R^{D\cdot D}$, learn two matrices on lower rank $U_{ij}\in \R^{D\cdot r}$ and $V_{ij}\in \R^{r*D}$ with $r<<D$ :
    $$ W_{ij} \approx U_{ij}\cdot V_{ij}$$
    The complexity after this would therefore be $O(N^² \cdot 2D\cdot r)$, a considerable improvement ! But now we need to see the results.
5) Olliver-Ricci curvature. Seen on a video, measures congestion. Need to dig that, would be interesting. Small explanation : in crisis, everyone rush on safe-assets (topological congestion). We could inject the curvature as a new feature in the HMM besides $H¹$ ?
6) Martin Hairer (Fields, 2014) : Path Signatures to FSLPredictor instead of gross return
7) Interesting thing I've seen too on HMM (from my finance club : LSM Investment Club) : Hierarchical Dirichlet Process (HDP) ! For now, we hardcode the number of hidden states (Bull, Normal, Volatil, Crisis). But is this right ? -> HDP learns the number of necessary states ! Is there a PyTorch that does this ?
8) Adversarial Training (Best ML course, LELEC2870). Interesting to apply it here ?
9) RL for leverage (for now, if("BULL"): leverage=1.5) to learn what is the best leverage. See paper on RL in finance (PPO agent);

## Bibliography and notes

A lot of work comes from the paper "Sheaf Cohomology of Linear Predictive Coding Networks" from Jeffrey Seely at Sakana AI (14 nov. 2025). My work is the implementation and some improvements of its work.
Moreover, LLM were used to generate code. This helps to go faster in my research. I check what the LLM does, but my competences and checks are not perfect. Some errors may have slipped in there. I'm constantly improving the methods used, and fixing bugs.

English is not my primary language, so feel free to correct any mistakes !

Also, do not hesitate to make a P-R with new methods, comments, better approach, etc ! I'm open to any improvement of FSL :) Don't hesitate to contact me : <mailto:arnullens@gmail.com> to discuss or if you have any idea !

Thank you for your interest in my project !
