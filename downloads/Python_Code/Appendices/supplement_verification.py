"""Deterministic numerical controls for supplements 07,08,10,11,12."""
import json
import math
from pathlib import Path
from fractions import Fraction
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
DT=torch.float64
results={}


def close(value, expected, atol=1e-10):
    if isinstance(value,torch.Tensor):
        value=value.detach().item()
    if isinstance(expected,torch.Tensor):
        expected=expected.detach().item()
    assert abs(float(value)-float(expected)) < atol,(value,expected)


def sequence():
    gradients=[]
    for detach_step in [None,2,3]:
        w=torch.tensor(.5,dtype=DT,requires_grad=True)
        u=torch.tensor(1.,dtype=DT,requires_grad=True)
        h=torch.tensor(0.,dtype=DT)
        for t in range(1,5):
            h=w*h+u
            if t==detach_step:
                h=h.detach()
        loss=(h-1).square()/2
        loss.backward()
        gradients.append([w.grad.item(),u.grad.item()])
    assert gradients==[[2.40625,1.640625],[2.1875,1.3125],[1.53125,.875]]
    g=torch.tensor([3.,4.],dtype=DT)
    clipped=g*min(1.,2./torch.linalg.vector_norm(g).item())
    assert torch.allclose(clipped,torch.tensor([1.2,1.6],dtype=DT))
    theta=torch.tensor([1.,-1.],dtype=DT)-.1*clipped
    assert torch.allclose(theta,torch.tensor([.88,-1.16],dtype=DT))
    state=0.
    for x in [1.,1.,-1.,-1.]:
        state=.5*state+x
    close(state,-1.125)
    results['07']=dict(bptt_gradients=gradients,clipped=clipped.tolist(),theta=theta.tolist(),wrong_carried_state=state)


def calibration():
    p=torch.tensor([.9,.8,.7,.6,.4,.3,.2,.1],dtype=DT)
    y=torch.tensor([1.,0.,1.,0.,1.,0.,0.,1.],dtype=DT)
    calibrated=torch.sigmoid(torch.logit(p)/2)
    def ece(prob):
        conf=torch.maximum(prob,1-prob)
        correct=((prob>=.5)==y.bool()).to(DT)
        total=0.
        for lower,upper in [(.5,.75),(.75,1.0001)]:
            selected=(conf>=lower)&(conf<upper)
            if selected.any():
                total += selected.to(DT).mean().item()*abs(conf[selected].mean()-correct[selected].mean()).item()
        return total
    close(ece(p),.25)
    close((p-y).square().mean(),.3)
    close(((p>=.5)==y.bool()).to(DT).mean(),.5)
    assert torch.equal(p>=.5,calibrated>=.5)
    positives=[.8,.6,.4]
    negatives=[.6,.3,.1]
    auc=sum(1. if a>b else .5 if a==b else 0. for a in positives for b in negatives)/9
    close(auc,5/6)
    posterior_odds=(.8/.2)*(.1/.9)/(.5/.5)
    posterior=posterior_odds/(1+posterior_odds)
    close(posterior,4/13)
    multi=torch.tensor([[.6,.2,.2],[.55,.449,.001]],dtype=DT)
    multi_new=torch.sqrt(multi)/torch.sqrt(multi).sum(dim=1,keepdim=True)
    assert multi[0,0]>multi[1,0] and multi_new[0,0]<multi_new[1,0]
    assert torch.equal(multi.argmax(dim=1),multi_new.argmax(dim=1))
    results['08']=dict(temperature2=calibrated.tolist(),ece_before=ece(p),ece_after=ece(calibrated),
                       brier_before=(p-y).square().mean().item(),brier_after=(calibrated-y).square().mean().item(),
                       auc=auc,posterior_after_prior_shift=posterior,multiclass_reversal=multi_new.tolist())


def masks():
    probability=torch.tensor([.8,.2,.6,.1],dtype=DT)
    logits=torch.logit(probability).requires_grad_()
    y=torch.tensor([1.,0.,1.,0.],dtype=DT)
    valid=torch.tensor([True,True,False,False])
    p=torch.sigmoid(logits)[valid]
    target=y[valid]
    bce=F.binary_cross_entropy_with_logits(logits[valid],target)
    dice=2*(p*target).sum()/(p.sum()+target.sum())
    loss=bce+.5*(1-dice)
    loss.backward()
    expected=torch.tensor([-.148,.132,0.,0.],dtype=DT)
    assert torch.allclose(logits.grad,expected,atol=1e-12,rtol=0)
    close(bce,-math.log(.8))
    close(dice,.8)
    empty_p=torch.tensor([.2,.3],dtype=DT,requires_grad=True)
    empty_y=torch.zeros(2,dtype=DT)
    empty_loss=1-2*(empty_p*empty_y).sum()/(empty_p.sum()+empty_y.sum())
    empty_loss.backward()
    assert torch.equal(empty_p.grad,torch.zeros(2,dtype=DT))
    iou_results=[]
    for a_value,expected_iou,expected_gradient in [(1.5,.75,-.5),(2.5,.8,.32)]:
        a=torch.tensor(a_value,dtype=DT,requires_grad=True)
        intersection=2*torch.minimum(a,torch.tensor(2.,dtype=DT))
        union=2*a+4-intersection
        iou=intersection/union
        (1-iou).backward()
        close(iou,expected_iou)
        close(a.grad,expected_gradient)
        iou_results.append(dict(a=a_value,iou=iou.item(),gradient=a.grad.item()))
    results['10']=dict(mask_bce=bce.item(),dice=dice.item(),combined_logit_gradient=logits.grad.tolist(),
                       empty_dice_gradient=empty_p.grad.tolist(),iou=iou_results,
                       average_pixel=.26,average_image=.5)


def regions():
    grid=torch.tensor([[1.,2.,4.],[3.,5.,8.],[6.,9.,13.]],dtype=DT,requires_grad=True)
    def bilinear(x,y):
        ix=int(torch.floor(x).item())
        iy=int(torch.floor(y).item())
        u=x-ix
        v=y-iy
        return (1-u)*(1-v)*grid[iy,ix]+u*(1-v)*grid[iy,ix+1]+(1-u)*v*grid[iy+1,ix]+u*v*grid[iy+1,ix+1]
    x=torch.tensor(1.25,dtype=DT,requires_grad=True)
    y=torch.tensor(.5,dtype=DT,requires_grad=True)
    value=bilinear(x,y)
    value.backward()
    close(value,4.125)
    close(x.grad,2.5)
    close(y.grad,3.25)
    assert torch.allclose(grid.grad,torch.tensor([[0.,.375,.125],[0.,.375,.125],[0.,0.,0.]],dtype=DT))
    other=bilinear(torch.tensor(.25,dtype=DT),torch.tensor(.5,dtype=DT))
    close(other,2.375)
    close((value+other)/2,3.25)
    close(x.grad/4,.625)
    close(y.grad/4,.8125)
    close(90/110,9/11)
    labels=[True,False,True,False,True]
    tp=0
    points=[]
    for index,label in enumerate(labels,1):
        tp+=int(label)
        points.append([Fraction(tp,3),Fraction(tp,index)])
    area=Fraction(0)
    previous=Fraction(0)
    for index,(recall,precision) in enumerate(points):
        if recall>previous:
            area+=(recall-previous)*max(row[1] for row in points[index:])
            previous=recall
    assert area==Fraction(34,45)
    results['11']=dict(bilinear=value.item(),gradient_x=x.grad.item(),gradient_y=y.grad.item(),
                       point2=other.item(),mean_two_points=3.25,nms_iou=90/110,ap=float(area))


def policies():
    gamma=.9
    transition=torch.tensor([[0.,1.],[0.,.5]],dtype=DT)
    reward=torch.tensor([0.,.5],dtype=DT)
    value=torch.linalg.solve(torch.eye(2,dtype=DT)-gamma*transition,reward)
    assert torch.allclose(value,torch.tensor([9/11,10/11],dtype=DT),atol=1e-12,rtol=0)
    horizon=[]
    old_b=0.
    for steps in range(1,5):
        a=max(.6,old_b)
        new_b=.5+.5*old_b
        horizon.append(dict(steps=steps,a=a,b=new_b,detour=old_b))
        old_b=new_b
    assert [x['a'] for x in horizon]==[.6,.6,.75,.875]
    close(1/(1-.5),2)
    close(.9**(3-1),.81)
    approximation=torch.tensor([.81,.9],dtype=DT)
    backup=torch.tensor([max(.6,.9*approximation[1].item()),.5+.45*approximation[1].item()],dtype=DT)
    residual=(backup-approximation).abs().max().item()
    close(residual,.005)
    bound=residual/(1-gamma)
    assert (value-approximation).abs().max().item()<=bound
    expected=.9*.8+.1*.2
    qlearning=.4+.5*(.1+.9*.8-.4)
    expected_sarsa=.4+.5*(.1+.9*expected-.4)
    sampled_sarsa=.4+.5*(.1+.9*.2-.4)
    close(qlearning,.61)
    close(expected_sarsa,.583)
    close(sampled_sarsa,.34)
    results['12']=dict(value=value.tolist(),horizons=horizon,residual=residual,error_bound=bound,
                       qlearning=qlearning,expected_sarsa=expected_sarsa,sampled_sarsa=sampled_sarsa)


sequence()
calibration()
masks()
regions()
policies()
output=dict(status='PASS',results=results)
Path(__file__).with_suffix('.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(output,ensure_ascii=False,indent=2))
