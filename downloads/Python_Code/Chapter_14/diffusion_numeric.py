"""Числовой прямой диффузионный процесс и малый тест шумопредсказания."""
import torch
from torch import nn

def forward_numeric():
    x0=torch.tensor([.5,-.5],dtype=torch.float64)
    noise=torch.tensor([1.,-1.],dtype=torch.float64)
    alpha_bar=.9*.8
    xt=alpha_bar**.5*x0+(1-alpha_bar)**.5*noise
    recovered=(xt-(1-alpha_bar)**.5*noise)/alpha_bar**.5
    predicted=torch.tensor([.8,-.7],dtype=torch.float64,requires_grad=True)
    loss=(predicted-noise).square().mean()
    loss.backward()
    assert torch.allclose(predicted.grad,torch.tensor([-.2,.3],dtype=torch.float64))
    return {"alpha_bar":alpha_bar,"x_t":xt,"recovered":recovered,
            "noise_mse":float(loss.detach()),"gradient":predicted.grad.clone()}

def tiny_denoising(steps=80):
    generator=torch.Generator().manual_seed(2026)
    torch.manual_seed(2026)
    x0=torch.randn(256,2,generator=generator)
    noise=torch.randn(256,2,generator=generator)
    alpha_bar=.72
    xt=alpha_bar**.5*x0+(1-alpha_bar)**.5*noise
    inputs=torch.cat([xt,torch.full((len(xt),1),.5)],dim=1)
    model=nn.Sequential(nn.Linear(3,32),nn.SiLU(),nn.Linear(32,2))
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    before=float((model(inputs)-noise).square().mean().detach())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss=(model(inputs)-noise).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    after=float((model(inputs)-noise).square().mean().detach())
    return {"before":before,"after":after,"steps":steps,
            "samples":len(inputs),"parameters":sum(p.numel() for p in model.parameters())}

if __name__=="__main__":
    torch.set_num_threads(2)
    report=forward_numeric()
    print("alpha_bar",report["alpha_bar"],"x_t",report["x_t"].tolist())
    print("recovered",report["recovered"].tolist(),"noise_mse",report["noise_mse"],
          "gradient",report["gradient"].tolist())
    result=tiny_denoising()
    assert result["after"]<result["before"]
    print("CPU wiring test",result)
