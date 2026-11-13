"""Малые воспроизводимые расчёты диагностики распределений без фотографий."""
import math
import torch

def rbf_mmd(x,y,sigma=1.):
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1]:
        raise ValueError("Ожидаются две матрицы с общей шириной признаков")
    if min(len(x),len(y))<2 or sigma<=0:
        raise ValueError("Нужны минимум два объекта и sigma > 0")
    kernel=lambda a,b:torch.exp(-torch.cdist(a,b).square()/(2*sigma*sigma))
    xx,yy,xy=kernel(x,x),kernel(y,y),kernel(x,y)
    biased=xx.mean()+yy.mean()-2*xy.mean()
    n,m=len(x),len(y)
    unbiased=((xx.sum()-xx.diag().sum())/(n*(n-1))+
              (yy.sum()-yy.diag().sum())/(m*(m-1))-2*xy.mean())
    return {"biased":float(biased),"unbiased":float(unbiased)}

def numeric_metrics():
    x=torch.tensor([[0.],[1.]],dtype=torch.float64)
    y=torch.tensor([[0.],[.5]],dtype=torch.float64)
    result=rbf_mmd(x,y)
    sx=float(x.std(unbiased=False));sy=float(y.std(unbiased=False))
    frechet=float((x.mean()-y.mean()).square())+(sx-sy)**2
    group_a=[.01,.04,.09];group_b=[.16]
    image_mean=sum(group_a+group_b)/4
    identity_mean=(sum(group_a)/len(group_a)+sum(group_b)/len(group_b))/2
    return {**result,"frechet_1d_population":frechet,
            "image_mean":image_mean,"identity_mean":identity_mean,
            "kernel_scale":1.,"reference_count":2,"sample_count":2}

if __name__=="__main__":print(numeric_metrics())
