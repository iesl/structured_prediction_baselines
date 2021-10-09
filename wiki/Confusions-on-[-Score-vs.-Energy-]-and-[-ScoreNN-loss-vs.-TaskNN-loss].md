### Score vs. Energy

Usual energy-based models are trained with lowering -E. 
In all our formulations, scoreNN outputs S which is -E. 
And therefore, for any loss function that had the form L(E) is replaced with L(-s) in our case. 

### Score moel vs. Energy model 

The prior description is a less confusing part, but perhaps a more confusing part is that we use the same scoreNN structure with [SEPN 2016, Belanger & McCallum ]([https://arxiv.org/abs/1511.06350]). In SPEN, the network outputs +E and we are outputting -E. However, this does not really cause a problem in terms of optimization as simply weight parameters can take opposite signs. 

### ScoreNN loss vs. TaskNN loss 

The taskNN logits are by themselves not normalized. In order to have the interpretation of a probability value per label, we have to normalize them with a sigmoid function. This normalized probability vector is input to our score and as tasks outputs this probability vector, if we could just work on a normalized probability vector, life would be easier for us. Nonetheless, as pytorch gradient caculation is numerically more stable when we directly take derivatives on logit, in case of training taskNN we use unnormalized logit as an input to taskNN loss whereas scoreNN takes normalzied probability vector as an input to its network. 

In summary, taskNN loss takes logit vector as an input and scoreNN takes probability vector as an input. 


