"""Проверки многошаговой цели, Huber и весов выборки."""
import torch

def n_step_target(rewards,gamma,next_value,terminated):
    result=sum((gamma**k)*float(r) for k,r in enumerate(rewards))
    if not terminated:result+=(gamma**len(rewards))*next_value
    return result

def weighted_huber(prediction,target,weights):
    residual=target.detach()-prediction
    absolute=residual.abs()
    losses=torch.where(absolute<=1,.5*residual.square(),absolute-.5)
    return (weights.detach()*losses).sum()/weights.detach().sum()

def numeric_report():
    q=torch.tensor([0.,0.,0.],dtype=torch.float64,requires_grad=True)
    target=torch.tensor([.5,-1.5,3.],dtype=torch.float64)
    weights=torch.tensor([.2,.5,1.],dtype=torch.float64)
    loss=weighted_huber(q,target,weights)
    loss.backward()
    losses=torch.tensor([.125,1.,2.5],dtype=torch.float64)
    probability=torch.tensor([.1,.3,.6],dtype=torch.float64)
    corrected=(probability*losses/(3*probability)).sum()
    assert torch.allclose(corrected,losses.mean())
    return {"three_step_continuing":n_step_target([1,0,-1],.9,2,False),
            "three_step_terminal":n_step_target([1,0,-1],.9,2,True),
            "weighted_huber":float(loss.detach()),
            "gradient":q.grad.tolist(),
            "uniform_expectation":float(losses.mean()),
            "corrected_expectation":float(corrected),
            "importance_product":(.8/.5)*(.2/.5)}

if __name__=="__main__":print(numeric_report())
