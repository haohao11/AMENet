# AMENET

Trajectory prediction is critical for applications of planning safe future movements and remains challenging even for the next few seconds in urban mixed traffic. The trajectory of a road user agent (e.g., pedestrian, cyclist and vehicle) is affected by the various behaviors of its neighboring agents in varying environments. Hence, its future trajectory is full of uncertainty and can be multi-modality.  To this end, we propose an end-to-end generative model named \emph{Attentive Maps Encoder Network (AMENet)} for accurate and realistic multi-path trajectory prediction. Our method leverages the target agent's motion information and the interaction information with the neighboring agents at each step, which is encoded as dynamic maps that are centralized on the target agent. A conditional variational auto-encoder module is trained to learn the latent space of possible future paths based on the dynamic maps and then used to predict multiple plausible future trajectories conditioned on the observed past trajectories. Our method reports the state-of-the-art performance (final/mean average displacement (FDE/ADE) errors 1.183/0.356 meters) on benchmark datasets and wins the first place in the open challenge of Trajnet (up to 08/2020)).

![AMENET](https://github.com/haohao11/AMENET/blob/master/model_framework.png)

Please cite:

```html
@article{cheng2020amenet,
  title={AMENet: Attentive Maps Encoder Network for Trajectory Prediction},
  author={Cheng, Hao and Liao, Wentong and Yang, Michael Ying and Rosenhahn, Bodo and Sester, Monika},
  journal={arXiv preprint arXiv:2006.08264},
  year={2020}
}
```
