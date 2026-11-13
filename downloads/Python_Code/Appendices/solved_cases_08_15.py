"""Eight alternative solved exercises. Small deterministic CPU computations only."""
from pathlib import Path
import json
import math
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
DT=torch.float64
results={}

def close(x,y,atol=1e-10):
    if isinstance(x,torch.Tensor):x=x.detach().item()
    assert abs(float(x)-float(y))<atol,(x,y)

old_prior=.25
new_prior=.5
def posterior(prior,positive_likelihood,negative_likelihood):
    return prior*positive_likelihood/(prior*positive_likelihood+(1-prior)*negative_likelihood)
old=[posterior(old_prior,.6,.3),posterior(old_prior,.4,.7)]
new=[posterior(new_prior,.6,.3),posterior(new_prior,.4,.7)]
close(old[0],.4);close(old[1],.16);close(new[0],2/3);close(new[1],4/11)
close(.375*old[0]+.625*old[1],old_prior)
close(.45*new[0]+.55*new[1],new_prior)
results['08']=dict(old=old,new=new,old_feature_probability=[.375,.625],new_feature_probability=[.45,.55],
                   risk_positive=3*(1-new[0]),risk_negative=new[0],cost_threshold=.75)

greedy_prob=.55*.5*.65
alternative_prob=.35*.9
greedy_nll=-math.log(greedy_prob)
alternative_nll=-math.log(alternative_prob)
assert alternative_prob>greedy_prob
unfinished_prefixes=[.55*.5,.55*.2,.35*.05,.35*.05]
assert max(unfinished_prefixes)<alternative_prob
assert max(.55*.3,.1)<alternative_prob
results['09']=dict(greedy_probability=greedy_prob,alternative_probability=alternative_prob,
                   greedy_nll=greedy_nll,alternative_nll=alternative_nll,
                   greedy_mean_nll=greedy_nll/3,alternative_mean_nll=alternative_nll/2,
                   greedy_perplexity=math.exp(greedy_nll/3),alternative_perplexity=math.exp(alternative_nll/2))

initial=torch.tensor([.1,.2,.4],dtype=DT)
logits=torch.logit(initial).requires_grad_()
p=torch.sigmoid(logits)
empty=torch.zeros(3,dtype=DT)
epsilon=1.
dice=(2*(p*empty).sum()+epsilon)/(p.sum()+empty.sum()+epsilon)
bce=F.binary_cross_entropy_with_logits(logits,empty)
combined=bce+.25*(1-dice)
combined.backward()
expected=initial/3+.25*initial*(1-initial)/(1.7**2)
assert torch.allclose(logits.grad,expected,atol=1e-12,rtol=0)
close(dice,10/17)
results['10']=dict(dice=dice.item(),dice_loss=1-dice.item(),bce=bce.item(),combined=combined.item(),
                   logit_gradients=logits.grad.tolist(),dice_probability_gradient=1/(1.7**2))

original=[10.,12.,34.,36.]
scaled=[(original[0]-6)*2,(original[1]-4)*1.5,(original[2]-6)*2,(original[3]-4)*1.5]
flipped=[80-scaled[2],scaled[1],80-scaled[0],scaled[3]]
assert scaled==[8.,12.,56.,48.]
assert flipped==[24.,12.,72.,48.]
anchor=[20.,10.,60.,50.]
def center_size(box):
    x1,y1,x2,y2=box
    return ((x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1)
ax,ay,aw,ah=center_size(anchor)
tx,ty,tw,th=center_size(flipped)
deltas=[(tx-ax)/aw,(ty-ay)/ah,math.log(tw/aw),math.log(th/ah)]
decoded_center=[ax+deltas[0]*aw,ay+deltas[1]*ah,aw*math.exp(deltas[2]),ah*math.exp(deltas[3])]
cx,cy,width,height=decoded_center
decoded=[cx-width/2,cy-height/2,cx+width/2,cy+height/2]
assert decoded==flipped
intersection=36*36
iou=intersection/(40*40+48*36-intersection)
close(iou,81/127)
results['11']=dict(scaled=scaled,flipped=flipped,deltas=deltas,decoded=decoded,iou=iou)

gamma=.8
v_b=.3/(1-.7*gamma)
v_a=max(.4,gamma*v_b)
close(v_b,15/22);close(v_a,6/11)
horizon=[]
old_b=0.
for remaining in range(1,6):
    horizon.append(dict(remaining=remaining,quick=.4,detour=old_b,best=max(.4,old_b)))
    old_b=.3+.7*old_b
close(horizon[2]['best'],.51)
results['12']=dict(v_b=v_b,v_a=v_a,gamma_switch=20/29,horizon=horizon)

online=torch.tensor([[1.,.5],[-.5,1.]],dtype=DT)
target=torch.tensor([[0.,1.],[1.,-.5]],dtype=DT)
feature=torch.tensor([2.,1.],dtype=DT)
online_q=online@feature
target_q=target@feature
tau=.2
updated=tau*online+(1-tau)*target
updated_q=updated@feature
choice=int(online_q.argmax())
goals=[.2+.95*target_q.max().item(),.2+.95*target_q[choice].item(),.2+.95*updated_q.max().item()]
def huber_and_gradient(goal):
    q=torch.tensor(.4,dtype=DT,requires_grad=True)
    loss=F.huber_loss(q,torch.tensor(goal,dtype=DT),delta=1.,reduction='sum')
    loss.backward()
    return dict(target=goal,error=goal-.4,loss=loss.item(),gradient=q.grad.item())
losses=[huber_and_gradient(goal) for goal in goals]
close(goals[0],1.625);close(goals[1],1.15);close(goals[2],1.435)
results['13']=dict(online_q=online_q.tolist(),target_q=target_q.tolist(),polyak=updated.tolist(),
                   polyak_q=updated_q.tolist(),losses=losses)

mu=torch.tensor([.5,-1.,.25],dtype=DT,requires_grad=True)
logvar=torch.log(torch.tensor([.25,1.,4.],dtype=DT)).requires_grad_()
noise=torch.tensor([-2.,.5,1.],dtype=DT)
z=mu+torch.exp(logvar/2)*noise
desired=torch.tensor([0.,0.,2.],dtype=DT)
reconstruction=.5*(z-desired).square().sum()
kl=.5*(mu.square()+torch.exp(logvar)-1-logvar).sum()
loss=reconstruction+.2*kl
loss.backward()
close(kl,1.78125);close(reconstruction,.28125);close(loss,.6375)
assert torch.allclose(mu.grad,torch.tensor([-.4,-.7,.3],dtype=DT),atol=1e-12,rtol=0)
assert torch.allclose(logvar.grad,torch.tensor([.175,-.125,.55],dtype=DT),atol=1e-12,rtol=0)
next_mu=mu.detach()-.05*mu.grad
next_logvar=logvar.detach()-.05*logvar.grad
next_z=next_mu+torch.exp(next_logvar/2)*noise
next_loss=.5*(next_z-desired).square().sum()+.2*.5*(next_mu.square()+torch.exp(next_logvar)-1-next_logvar).sum()
assert next_loss<loss.detach()
results['14']=dict(z=z.tolist(),kl=kl.item(),reconstruction=reconstruction.item(),loss=loss.item(),
                   gradient_mu=mu.grad.tolist(),gradient_logvar=logvar.grad.tolist(),next_mu=next_mu.tolist(),next_loss=next_loss.item())

generated=torch.tensor([[.2,.2],[.8,.2],[.5,.5]],dtype=DT)
train=torch.tensor([[.2,.2],[.7,.3],[.9,.9]],dtype=DT)
held=torch.tensor([[.1,.3],[.9,.1],[.4,.6]],dtype=DT)
def nearest(g,reference):
    return (g[:,None,:]-reference[None,:,:]).square().mean(dim=2).min(dim=1).values
train_min=nearest(generated,train)
held_min=nearest(generated,held)
pair=((generated[:,None,:]-generated[None,:,:]).square().mean(dim=2))[torch.triu(torch.ones(3,3,dtype=torch.bool),diagonal=1)]
augmented=nearest(generated,torch.cat([train,generated[2:3]],dim=0))
close(train_min.mean(),1/60);close(held_min.mean(),.01);close(pair.mean(),.12);close(augmented.mean(),1/300)
groups=[set(range(1,9)),set(range(9,11)),set(range(11,13))]
assert all(not groups[i]&groups[j] for i in range(3) for j in range(i+1,3))
results['15']=dict(train_min=train_min.tolist(),held_min=held_min.tolist(),train_mean=train_min.mean().item(),
                   held_mean=held_min.mean().item(),pairwise=pair.tolist(),pairwise_mean=pair.mean().item(),
                   augmented_train_min=augmented.tolist(),augmented_train_mean=augmented.mean().item(),groups=[len(x) for x in groups])
close(2*pair.sum()/9,.08)

output=dict(status='PASS',torch=str(torch.__version__),results=results)
Path(__file__).with_suffix('.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(output,ensure_ascii=False,indent=2))
