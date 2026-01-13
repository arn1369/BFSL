# Biomedical Functorial Sheaf Learning Networks

Last version : Alpha 0.0.1

Here we apply FSL to the health sector. For now, we use the MIMIC IV database. For confidentiality reasons, I can't publish the data itself : only the results will be visible here. The goal with FSL is to model physical interactions between organs to reconstruct missing data.

1. Why FSL ?
    Unlike other methods, FSL doesn't ignore structural relationships between variables. So we use Sheaf Cohomology to measure "consistency" (in other words, it checks that the predicted value make topological sense). Here, we have that ICU is full of missing data/holes. FSL uses the diffusion to propagate the information to predict what's in the holes. My goal by using FSL is to get biological (topological) coherence instead of statistical correlation.

    We have applied it here by using MIMIC-IV database. We extract (for now) 6 features (time-series) :
        - Heart Rate
        - Systolic Blood Pressure
        - Diastolic Blood Pressure
        - Oxygen Saturation
        - Respiratory Rate
        - Temperature
    This can sound like it's not enough features. But the thing is that we have a high complexity : $O(N²)$. So for now, I use this. I will probably increase this later, but I will need a better GPU.

    Here the goal of the FSL is signal reconstruction : we hide 20% to 50% chunks of data, and ask the model to reconstruct the missing curves. This is useful when a sensor has a problem (noisy, disconnected, ...).

2. Architure
    So here, the designed architecture is in 3 levels (because of compatibility with the basic FSL model). This will probably change in the future. For now we "model" :
        1) Level 0 : Fine ("Organs")
        2) Level 1 : Medium ("Clusters"). e.g Cardio-Pulmonary system
        3) Level 2 : Coarse ("Patient states")

    And the model diffuses information up and down this hierarchy. E.g If the HR is missing, the model looks at the BP and the Patient State to deduce what the HR should be.

    Loss : here we combine L1 loss (absolute error) and Gradient loss (shape/slope) to prevent model collapse (flat lines predictions). This will probably change in the future too, as there exists some better approaches (probably, need to be tested) : notably shape loss, dilate loss and quantile loss.

    Note on data, we use forward and backward filling data to avoid most possible NaN's (in Alpha \#0.0.1, we get 57 NaN's at the end instead of ~10k)

## Phase 1

    At the end of Phase 1 (implementing the basic model and signal reconstruction), we get some interesting results :
        - MSE of 0.3912 (normalized)
        - MAE of 0.4348 (normalized)

    Here is an approximation of the precision per vital sign :

        HeartRate       : Mean error 7.76 bpm (Norm: 0.388)
        SystolicBP      : Mean error 8.45 mmHg (Norm: 0.338)
        DiastolicBP     : Mean error 7.73 mmHg (Norm: 0.516)
        SpO2            : Mean error 1.76 % (Norm: 0.585)
        RespRate        : Mean error 2.13 bpm (Norm: 0.426)
        Temperature     : Mean error 0.54 °F (Norm: 0.358) 

    These are nice results. We have to consider that measurements are not perfect too. I need to check what is the standard error on measurements sensors to see what is nice or not, and what is biased (if the model has better prediction than the measurement error, there is a problem).

    Visually (see the plot in /visuals), we can see that the model reconstructs (approximately) nice results. Sometimes, we can see a sensor failure (e.g. the temperature is 0°F). The models choose to "ignore" this noise.

### Problems

    We had problems during development :
        - *model collapse* (flat lines predictions) of MSE. It was "fixed" with the L1 + Shape Loss, but it's not perfect. MSE is bad because it penalizes too much large errors. But even now, it's not nice enough : we have something like a moving average. The thing is that we won't be able to model succint crashes, as it can be caused from nearly anything. So the model needs a lot of data to be able to predict this. And with that complexity, it will be hard. I absolutely need to find a better solution on this. Or maybe the goal is a "general prediction" that describes a tendancy, but it is somewhat useless.

        - *"shaky" predictions* : the learning rate scheduler has been triggered too late. This will be fixed in \#Alpha 1.0.2.
        
        - *gaps on real data* : there is too much gaps/missing data in the dataset. We had to use ffill/bfill to give the model a nice learning. This is bad and I need to find a better solution in future versions. I need to avoid backward fill as it acts like a "look-ahead bias".

        - *Topological non-stationarity* : the relation between two features change if the patient is in "idle" or in "crisis" state. For now, we put global fix parameters, but this is bad. The model learns a "mean" relation, always valid but never precise. This will change in \#Alpha 1.0.3

### Improvements

    1) Fix the scheduler (\#Alpha 1.0.2)
    2) MultiModal integrations : Add lab results ("slow" features that can help find "fast" features)
    3) Use H¹ to predict sepsis or mortality ? high H¹ could tell an organ failure.
    4) It could be interesting too to add a "warning" to patients that have a high H¹. The goal is to say : "These patients are sick. I'm sure. These ones are Healthy. I'm sure. These ones, H¹ is higher than the norm. Go check a doctor.". This could probably prevent a lot of medical problems ! 
    5) Visualization of adjacency matrix (to see which "organs" are connected, according to the model). -> discovers the link between HR and BP or not ?

### Future Phase 1 versions

    \#Alpha 0.0.2 : 
        - Change the bfill
        - split/train by seperating patients ID.
        - reduce loss weight of diffusion
        * Implementation comments :
            -> model find high link between DiastolicBP and SpO2 -> wtf ?
            -> Problems of over-smoothing (scatter_pred)

    \#Alpha 0.0.3 :
        - Fix over-smoothing (probabilistic output -> normal distrib ?) other methods ?

## Phase 2

Try to use data that have different time scales

## Phase 3

Understanding how the patient has problems (what feature is the problem ?). Ex : two patients have tachycardy, but from different sources.
Use the cycles of H¹ to do this.

## Phase 4

What if I give this drug ? -> prediction of FSL on the patient
-> perturbate a node, and see how it diffuses.

## Bibliography and TO-DO list

So this work is an application of FSL, which is based on the super cool article of [Sheaf Cohomology of Linear Predictive Coding Networks](https://arxiv.org/pdf/2511.11092) from Jeffrey Seely at Sakana AI  (14 nov. 2025).

Also, LLM were used to generate some code. This is useful as it helps me to move faster in my research. But here, for privacy reasons, I had to take care with them, notably on the MIMIC-IV database. No LLMs has accessed any sort of data from MIMIC-IV. I mostly used LLMs to help me debug a lot of bugs (notably the wrong dimensions ones!). The code is always executed locally. They were used according to the "CITI Data or Specimens Only Research". I highly thanks PhysioNet and MIT to have access to this database. I hope this project will be useful for the people that needs it.

Don't hesitate to send me a mail [arnullens@gmail.com](mailto:arnullens@gmail.com), if you are curious, want to add it to your models, and if you have suggestions or improvements !

Thank you for your interest in my project !
