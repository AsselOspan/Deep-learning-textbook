"""Independent numerical checks for seven solved academic control cases."""
import argparse
from itertools import product
import json
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

D = torch.float64


def case_01():
    single = sum((prediction-target)**2 for prediction, target in product((2., 4.), (2., 4.)))/4
    ensemble = sum(((a+b)/2-target)**2 for a, b, target in product((2., 4.), repeat=3))/8
    correlated = sum(((a+a)/2-target)**2 for a, target in product((2., 4.), repeat=2))/4
    oracle = sum((3-target)**2 for target in (2., 4.))/2
    assert (single, ensemble, correlated, oracle) == (2., 1.5, 2., 1.)
    return dict(single=single, independent_ensemble=ensemble, correlated_ensemble=correlated, conditional_mean=oracle)


def case_02():
    hessian = torch.diag(torch.tensor([4., 1.], dtype=D))
    initial = torch.ones(2, dtype=D)
    loss = lambda theta: float(.5*theta @ hessian @ theta)
    rows = []
    for lr in (.1, .5, .6):
        theta = initial-lr*(hessian @ initial)
        rows.append(dict(lr=lr, theta=theta.tolist(), loss=loss(theta)))
    p = torch.diag(torch.tensor([.25, 1.], dtype=D))
    preconditioned = initial-.5*(p @ hessian @ initial)
    assert abs(loss(preconditioned)-.625)<1e-12
    assert abs(rows[0]['loss']-1.125)<1e-12 and abs(rows[2]['loss']-4)<1e-12
    return dict(initial_loss=loss(initial), rows=rows, preconditioned=preconditioned.tolist(), preconditioned_loss=loss(preconditioned), stable_upper_lr=.5)


def case_03():
    x = torch.arange(1., 20., dtype=D).reshape(1, 1, 19).requires_grad_()
    a = F.conv1d(x, torch.ones(1, 1, 5, dtype=D), stride=2, padding=2)
    b = F.conv1d(a, torch.ones(1, 1, 3, dtype=D), stride=2)
    c = F.conv1d(b, torch.ones(1, 1, 3, dtype=D), dilation=2, padding=2)
    first = torch.autograd.grad(c[0, 0, 0], x, retain_graph=True)[0]
    last = torch.autograd.grad(c[0, 0, -1], x)[0]
    first_indices = torch.nonzero(first.flatten()).flatten().tolist()
    last_indices = torch.nonzero(last.flatten()).flatten().tolist()
    assert first_indices == list(range(15)) and last_indices == list(range(2, 19))
    shape, receptive, jump, center = 19, 1, 1, .5
    geometry = []
    for kernel, stride, padding, dilation in ((5, 2, 2, 1), (3, 2, 0, 1), (3, 1, 2, 2)):
        shape = (shape+2*padding-dilation*(kernel-1)-1)//stride+1
        receptive += (kernel-1)*dilation*jump
        center += ((kernel-1)*dilation/2-padding)*jump
        jump *= stride
        geometry.append(dict(size=shape, receptive=receptive, jump=jump, center=center))
    assert list(c.shape)==[1, 1, 4] and geometry[-1]==dict(size=4,receptive=25,jump=4,center=2.5)
    return dict(geometry=geometry,first_input_indices=first_indices,last_input_indices=last_indices,weights=11)


def case_04():
    rows=[]
    for beta, value in product((1., 4.), (0., 1.)):
        x=torch.tensor(value,dtype=D,requires_grad=True)
        probability=torch.sigmoid(beta*x)
        derivative=torch.autograd.grad(probability,x)[0]
        loss=F.softplus(-beta*x)
        grad=torch.autograd.grad(loss,x,create_graph=True)[0]
        curvature=torch.autograd.grad(grad,x)[0]
        p=float(probability.detach())
        assert abs(float(derivative)-beta*p*(1-p))<1e-12
        assert abs(float(grad.detach())-beta*(p-1))<1e-12
        assert abs(float(curvature)-beta*beta*p*(1-p))<1e-12
        rows.append(dict(beta=beta,x=value,p=p,activation_derivative=float(derivative),loss=float(loss.detach()),gradient=float(grad.detach()),curvature=float(curvature)))
    return dict(rows=rows)


def case_05():
    result={}
    for label in ('zero','none'):
        p=nn.Parameter(torch.tensor(2.,dtype=D))
        optimizer=torch.optim.SGD([p],lr=.1,momentum=.9,weight_decay=.05)
        p.grad=torch.tensor(1.,dtype=D)
        optimizer.step()
        first=float(p.detach())
        p.requires_grad_(False)
        p.grad=torch.zeros((),dtype=D) if label=='zero' else None
        optimizer.step()
        result[label]=dict(first=first,second=float(p.detach()),buffer=float(optimizer.state[p]['momentum_buffer']))
    assert abs(result['zero']['second']-1.78155)<1e-12
    assert abs(result['none']['second']-1.89)<1e-12
    return result


def case_06():
    q=torch.tensor(1.,dtype=D,requires_grad=True)
    keys=torch.log(torch.tensor([1.,2.,3.],dtype=D)).requires_grad_()
    values=torch.tensor([1.,-1.,2.],dtype=D,requires_grad=True)
    scores=q*keys
    scores.retain_grad()
    weights=scores.softmax(0)
    context=weights @ values
    loss=.5*(context-.5).square()
    loss.backward()
    manual=torch.tensor([1/108,-11/54,7/36],dtype=D)
    torch.testing.assert_close(scores.grad,manual)
    torch.testing.assert_close(values.grad,torch.tensor([1/18,1/9,1/6],dtype=D))
    masked=weights.detach().clone();masked[2]=0
    correct=torch.softmax(scores.detach().masked_fill(torch.tensor([False,False,True]),-torch.inf),0)
    return dict(weights=weights.detach().tolist(),context=float(context.detach()),score_grad=scores.grad.tolist(),query_grad=float(q.grad),value_grad=values.grad.tolist(),masked_context=float(correct @ values.detach()),incorrect_postmask_context=float(masked @ values.detach()))


def case_07():
    a=torch.tensor(.6,dtype=D,requires_grad=True)
    h1=torch.tensor(1.,dtype=D)
    h2=a*h1-1
    h3=a*h2+2
    full_loss=.5*(h3-1).square()
    full_grad=torch.autograd.grad(full_loss,a)[0]
    truncated=.5*(a*h2.detach()+2-1).square()
    truncated_grad=torch.autograd.grad(truncated,a)[0]
    loss=lambda parameter:.5*(parameter*parameter-parameter+1)**2
    full_update=.6-.1*float(full_grad)
    truncated_update=.6-.1*float(truncated_grad)
    assert abs(float(full_grad)-.152)<1e-12 and abs(float(truncated_grad)+.304)<1e-12
    assert loss(full_update)<loss(.6)<loss(truncated_update)
    return dict(states=[1.,float(h2.detach()),float(h3.detach())],state_derivatives=[0.,1.,.2],loss=loss(.6),full_gradient=float(full_grad),truncated_gradient=float(truncated_grad),full_update=full_update,truncated_update=truncated_update,full_new_loss=loss(full_update),truncated_new_loss=loss(truncated_update))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    result={str(n):globals()[f'case_{n:02}']() for n in range(1,8)}
    text=json.dumps(result,ensure_ascii=False,indent=2)
    print(text)
    if args.json:args.json.write_text(text+'\n',encoding='utf-8')
    print('PASS: seven solved cases independently checked')


if __name__=='__main__':main()
