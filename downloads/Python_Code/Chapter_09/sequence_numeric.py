"""Потеря по действительным токенам, causal-mask и маленькое дерево декодирования."""
import math
import torch
import torch.nn.functional as F

def sequence_report():
    probabilities=torch.tensor([
        [.05,.8,.1,.05],[.1,.1,.5,.3],
        [.25,.25,.25,.25],[.25,.25,.25,.25]],dtype=torch.float64)
    logits=probabilities.log().detach().requires_grad_(True)
    targets=torch.tensor([1,2,3,0])
    valid=(targets!=0).sum()
    total=F.cross_entropy(logits,targets,ignore_index=0,reduction="sum")
    loss=total/valid
    loss.backward()
    scores=torch.eye(2,dtype=torch.float64)/(2**.5)
    forbidden=torch.triu(torch.ones(2,2,dtype=torch.bool),diagonal=1)
    weights=scores.masked_fill(forbidden,-torch.inf).softmax(-1)
    sequence_probabilities={"A EOS":.6*.51,"A C EOS":.6*.49,"B EOS":.4*.99,
                            "B C EOS":.4*.01}
    return {"loss":float(loss.detach()),"perplexity":math.exp(float(loss.detach())),
            "wrong_denominator_loss":float(total.detach()/len(targets)),
            "pad_gradient":logits.grad[-1].tolist(),
            "first_token_gradient":logits.grad[0].tolist(),
            "causal_first_weights":weights[0].tolist(),
            "causal_second_weights":weights[1].tolist(),
            "greedy_probability":sequence_probabilities["A EOS"],
            "best_probability":max(sequence_probabilities.values()),
            "best_sequence":max(sequence_probabilities,key=sequence_probabilities.get)}

if __name__=="__main__":
    torch.set_num_threads(2)
    print(sequence_report())
