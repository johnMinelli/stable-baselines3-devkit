# DO NOT CLIP STD!!
# DO NOT CLIP ACTION PRE-OPTIMIZATION!!

So the good thing we have is a more complex attention mechanism that could extract from the past more info to correct actions
this works on short terms i.e. retry with more wisdom -> long histories and retries have a complexity problem which should be avoided
the scene has to be in training set -> ow is not solvable because generalization requires a big network and a big training that should be avoided
then we need to train a robust-to-failure policy through retries by collecting demonstrations of retry until success -> we have to exclude cases of simple failure

**HISTORY TO FACE DIFFICULTIES OF THE TASK** -> *when the task is hard and even in normal conditions there are failures*, if a retrying policy fails it's because is not able to retry in a different manner: _
**HISTORY TO FACE RANDOMIZATION IN THE TASK** -> some actions are dependent to how the world respond (e.g. static objects, high density): _
**HISTORY CAN REPLACE PRECISE PERCEPTION**: -> the state can be not reliable (e.g. occlusions) act some steps with no cam


# Franklin (cluster) inference
                                                ## (policy device)cpu (simulation device)cuda
Successes: 0.12                                 Successes: 0.14                             
Mean reward: 30.33                              Mean reward: 33.17                          
Mean episode length: 178.70                     Mean episode length: 175.60                 
                                                ## cuda cuda    
Successes: 0.14                                 Successes: 0.12                             
Mean reward: 27.50                              Mean reward: 31.50                          
Mean episode length: 175.24                     Mean episode length: 178.64                 
                                                ## cpu cpu
Successes: 0.12                                 Successes: 0.14
Mean reward: 30.33                              Mean reward: 33.17
Mean episode length: 178.70                     Mean episode length: 175.60
                                                ## cuda cpu
Successes: 0.14                                 Successes: 0.12
Mean reward: 27.50                              Mean reward: 31.50
Mean episode length: 175.24                     Mean episode length: 178.64
---
# Local inference
                                                ## cpu cuda
Successes: 0.12                                 Successes: 0.12       
Mean reward: 35.97                              Mean reward: 35.97       
Mean episode length: 180.72                     Mean episode length: 180.72       
                                                ## cuda cuda                    
Successes: 0.04                                 Successes: 0.14                          
Mean reward: 29.73                              Mean reward: 32.93                       
Mean episode length: 193.08                     Mean episode length: 176.44              
                                                ## cpu cpu
Successes: 0.12                                 Successes: 0.12
Mean reward: 35.97                              Mean reward: 35.97
Mean episode length: 180.72                     Mean episode length: 180.72
                                                ## cuda cpu
Successes: 0.04                                 Successes: 0.14
Mean reward: 29.73                              Mean reward: 32.93
Mean episode length: 193.08                     Mean episode length: 176.44



- shared or same architecture for actor and critic
- finish StackPickMulti-v1 env
- clarify mem length requirements
- clarify army or not

Differences:
- terminal bootstrapping
- std as log

fix the mlp network
check isaac training on Franklin with hopefully correct parameters
finish rob11 installation checking isaac correct execution and code execution and wandb (main cause)

focus on recurrence of tokens, and mlp on top to extract the actions
the recurrent stores the goal overtime
for now we do only on step (single pass) the contains a single timestep but multiple tokens. For locomotion that would be enough and I want convergence with army or not army.
later recurrence multi step (multiple passes) (maybe also multi timesteps?). That is to exten recurrent reasoning though time without having quadratic explosion, army can be advantaged here? To help manipulation we could do restarts in random points of the MDP, since if you understood the goal the resolution can be achieved easier in this way
we take inspiration also from the methods of learning across MDPs of skild 

    
test number of epochs
test normalization in attention



test safe init
test unsafe init
test norm
test critic from transformer
plot gradient magnitude
plot entropy scores
plot residual magnitude
plot the Max Gradient (infinity norm). Sometimes the L2 norm looks fine, but one specific layer (usually Layer 0 or the Attention Output) has a massive spike that breaks the numerical stability.
plot Var(R) - If this drops to 0, your environment/agent interaction has become deterministic (dead agent). If this is high but explained_var is low, your Critic is failing to model a complex world.


 3267.02it/s] (s/epoch: 0.449s, s/rollout: 0.034s,   8 0
 2553.05it/s] (s/epoch: 0.795s, s/rollout: 0.044s,   8 8  --> 710
 1925.79it/s] (s/epoch: 1.244s, s/rollout: 0.045s    8 12  -> 860
 4511.16it/s] (s/epoch: 0.215s, s/rollout: 0.032s,   1 0
 3348.60it/s] (s/epoch: 0.589s, s/rollout: 0.036s,   1 1
 2656.43it/s] (s/epoch: 0.838s, s/rollout: 0.038s,   1 2
 2324.79it/s] (s/epoch: 1.052s, s/rollout: 0.037s,   1 3
 2031.57it/s] (s/epoch: 1.250s, s/rollout: 0.038s    1 4  -> 600
 
solved the big drop (curriculum of the env) and increased the negative reward for joints from the beginning
solved the spiking rewards and too uniform batches (random termination)
num of heads explored: increasing them improves up to a certain point because we reduce attention entropy but we reduce information
init explored: small init regulates the residual ratio and this is really important because MLP optimization is faster, we can then learn around the mlps with attention residual

interesting experiment is about the value loss evolution: [taking](https://wandb.ai/johnminelli/army/runs/w4mvfz9k?nw=nwuserjohnminelli) a trained mlp value net and trying to train an mlp policy with frozen value net. it [fails](https://wandb.ai/johnminelli/army/runs/grwtekbv?nw=nwuserjohnminelli).
I then tried to [unlock](https://wandb.ai/johnminelli/army/runs/yppa12qt?nw=nwuserjohnminelli) it and train a policy and is fine. The so fine-tuned mlp value net has been tested again for frozen training and it is [failing](https://wandb.ai/johnminelli/army/runs/43x373nv?nw=nwuserjohnminelli). aka the value net has to evolve with the policy.

I wanted to implement GRPO || value estimation in transformers has problems 
Don't like entropy tuning so use the action spread
optimistic makes sense
MLP good results

value loss too smooth we don't have enough balance in loss terms (value loss should be lowered): at the condition of enough entropy, policy update means also choosing new directions that makes the value loss increase, if this does not happen there is a problem
entropy is a secondary signal. Performance are affected by value and policy. Those are affected by entropy. if something is wrong we see effects in value and policy. Something is breaking them. Such 'something' can be the entropy.
kl too high means the learning rate is too high (indeed adaptive lr tries to reduce it) and as consequence the policy update is clipped and we are moving in not safe policies (lr high pushes also the entropy to decrease)


tests coverage
processed observation in the conf file
test install from 0
push
switch and align again
- make the recurrent works
- check the ACTransformer if it can be pushed
