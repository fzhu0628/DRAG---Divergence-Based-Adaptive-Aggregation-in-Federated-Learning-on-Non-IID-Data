# DRAG-Divergence-based-adaptive-aggregation-in-federated-learning-on-non-iid-data
## About this paper
- This paper revolves around the topic of mitigating the _**client-drift**_ effect in federated learning (FL), which is a joint consequence of
  * **Heterogeneous data distributions** across agents/clients/workers.
  * **Intermittent communication** between the central server and the agents.
- In light of this, we develop a novel algorithm called _divergence-based adaptive aggregation (**DRAG**)_ which
  - Hinges on the metric introduced as "**degree of divergence**" that quantifies the angle between the local gradient direction of each agent and the _reference direction_ (the one we desire to move toward).
  - Dynamically "drags" the received local updates toward the reference direction in each round _without extra communication overhead_.
- Rigorous **convergence analysis** is provided for DRAG, proving a **sublinear** convergence rate.
- Experiments demonstrate the superior performance of DRAG compared to advanced algorithms.
- Additionally, DRAG exhibits remarkable resilience towards byzantine attack thanks to its "dragging" nature. Numerical results are provided to showcase this property.
## Motivations
### Client-Drift Effect
- In the context of FL, the classical problem of interest is to minimize the average/sum of the local loss functions across agents/clients. With a predominant probability, different agents have distinct local datasets generated from different distributions, thus having distinct local loss functions, which is where the "**_heterogeneity_**" comes from.
- On a different note, since communication overhead is a bottleneck in FL, agents cannot afford to talk to the server each time they have finished computing a gradient. Ergo, the famous federated averaging (FedAvg) algorithm requires that each agent compute multiple local updates on its local model before communicating with the server to upload its latest model. This is known as the universal "**_intermittent communication_**" mechanism in FL.

![image](https://github.com/user-attachments/assets/4e282bae-b369-4abe-ba1b-5b8c476259cf)

This figure illustrates the client-drift effect (source: Karimireddy et al., 2021).

However, if these two features are combined together, the "client-drift" effect would then arise since each agent inevitably _updates its local model towards its local optimum_. Furthermore, simply adding these updated models up and averaging them will not necessarily perfectly cancels the heteregeneity out, but will impede the convergence rate, as both theoretically and experimentally demonstrated by previous work.
### Byzantine Attacks
As FL operates in a distributed manner, it is susceptible to adversarial attacks launched by malicious clients, commonly known as Byzantine attacks. Classical Byzantine attacks include
- Reversing the direction of the local gradient.
- Scaling the local gradient by a factor to negatively influence the training process.

![image](https://github.com/user-attachments/assets/a1bd598b-a2ab-418b-b021-25dbc2be1201)

This figure illustrates the Byzantine attack.
## Exhibition of the DRAG algorithm
### Problem Formulation
- Problem of interest:

![image](https://github.com/user-attachments/assets/c9c2c79d-3fae-4beb-aaca-bc928e6ae7d6)

- Local update formula of each agent:

![image](https://github.com/user-attachments/assets/3bfb7901-92fe-4857-b16b-3c79c9dd6de2)

- Model difference uploaded by each agent $m$ at round $t$:

![image](https://github.com/user-attachments/assets/02b64e10-1103-4a80-82ce-9cf6560ce714)

- Server aggregation:

![image](https://github.com/user-attachments/assets/60d070f6-9460-4057-a3e9-cfd258dfe3c9)

- Server aggregation (w/ Byzantine attacks):

![image](https://github.com/user-attachments/assets/1e751fa5-500c-46c2-8bbd-14570c956f28)

Here, $\hat{\mathbf{g}}_m^t=p_m^tg_m^t$ is the perturbed gradient where $p_m^t$ can be either positive or negative, and $A^t$ denotes the set of malicious agents.
### Key Definitions
1. **Reference direction**: The objective of the reference direction is to offer a practical and sensible direction for modifying the local gradients, denoted as $r^t$

![image](https://github.com/user-attachments/assets/2531da0a-2b6a-4ff7-8ab3-c5bcc822e0eb)

where 

![image](https://github.com/user-attachments/assets/6e63f78a-d588-439b-89d1-a818bf741169)

Here, $v_m^t$ is the modified model difference for agent $m$ at round $t$ (defined later). The reference direction takes a _bootstrapping form_ and is a _weighted sum of all the historical global update directions_.

2. **Degree of divergence**: This is the fundamental concept of the DRAG algorithm, measuring the extent to which the local update direction diverges from the reference direction:

![image](https://github.com/user-attachments/assets/40ff0b7a-552a-4833-a301-0352a59a829a)

where the constant $c$ helps provide more flexibility.

3. **Vector manipulation**: Using the reference direction and degree of divergence. we "**drag**" each local model difference towards the reference function via vector manipulation:

![image](https://github.com/user-attachments/assets/2a89390e-be55-44ac-bc41-e79a53dbb679)

Note that we normalize the reference direction such that the modified difference consistently has a greater component aligned with $r^t$ compared to $g_m^t$.

![image](https://github.com/user-attachments/assets/815c89e2-2322-4519-8f2f-b67026412f1b)

### Algorithm Description
1. Server broadcasts the global parameter to a selected set of agents at round $t$.
2. Each agent performs $U$ local updates and sends the model difference to the server.
3. Server calculates the reference direction and the degree of divergence.
4. Server modifies the model difference and aggregates the modified differences.

### Defending Against Byzantine Attacks
To deal with Byzantine attacks, we need to make adaptations to the DRAG algorithm. Since the malicious attacks can undermine the effectiveness of the reference direction, we update the definition of reference direction in this setup. 

**Reference direction for Byzantine attacks**: The server maintains a small root dataset $D_{root}$. At each round $t$, the server also updates a copy of the current global model for $U$ iterations using the root dataset:

![image](https://github.com/user-attachments/assets/1e4a3788-7c76-44d0-8122-737700d3edd8)

The reference direction is then defined as

![image](https://github.com/user-attachments/assets/4401f67a-3714-4e4f-88dd-3075505e6a07)

We also need to update the vector manipulation procedure in this case, since the malicious agents might scale the module of the local model differences.

**Vector manipulation for Byzantine attacks**: The modified model difference is defined as

![image](https://github.com/user-attachments/assets/e9e37d24-749c-48c7-a83c-bb6977b64ff2)

## Convergence Analysis
Assuming each local loss function is smooth and lower-bounded, and assuming that the local gradient estimator is unbiased and has bounded variance, we show that DRAG converges to the first-order stationary point (FOSP) at a sublinear rate:

![image](https://github.com/user-attachments/assets/6ed0b694-096c-44de-a7cb-89af913d4dc2)

## Numerical Results
### Client-Drift Mitigation
We compare our DRAG algorithm with advanced algorithms on the CIFAR-10 dataset:

![image](https://github.com/user-attachments/assets/e783c64b-e645-43f3-9f57-a8eb4c954286)


### Byzantine Attack
We compare DRAG with other algorithms under Byzantine attacks with high data heterogeneity:

![image](https://github.com/user-attachments/assets/6a9bb515-f342-43cb-879c-5bb4054a9bcf)

