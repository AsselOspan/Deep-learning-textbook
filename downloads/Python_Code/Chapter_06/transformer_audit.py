"""Проверяемый компактный patch-Transformer и переносимый checkpoint."""
from pathlib import Path
import torch
from torch import nn

def attention_numeric():
    q=torch.eye(2,dtype=torch.float64)
    k=q.clone()
    v=torch.tensor([[2.,0.],[0.,4.]],dtype=torch.float64)
    scores=q@k.T/(q.shape[-1]**.5)
    weights=scores.softmax(-1)
    return {"scores":scores,"weights":weights,"context":weights@v}

class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch=nn.Conv2d(3,16,kernel_size=8,stride=8)
        self.position=nn.Parameter(torch.zeros(1,16,16))
        layer=nn.TransformerEncoderLayer(
            d_model=16,nhead=4,dim_feedforward=32,
            dropout=0.,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(
            layer,num_layers=1,enable_nested_tensor=False)
        self.head=nn.Linear(16,3)

    def forward(self,images):
        if images.ndim!=4 or tuple(images.shape[1:])!=(3,32,32):
            raise ValueError("Ожидается [B,3,32,32]")
        tokens=self.patch(images).flatten(2).transpose(1,2)
        encoded=self.encoder(tokens+self.position)
        return self.head(encoded.mean(dim=1))

def checkpoint_roundtrip(model,inputs,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    model.eval()
    with torch.no_grad():
        expected=model(inputs)
    torch.save({"state_dict":model.state_dict(),
                "architecture":"PatchEncoder",
                "input_shape":[3,32,32],"seed":2026},path)
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    if checkpoint["architecture"]!="PatchEncoder":
        raise ValueError("Другая архитектура checkpoint")
    restored=PatchEncoder().eval()
    restored.load_state_dict(checkpoint["state_dict"],strict=True)
    with torch.no_grad():
        actual=restored(inputs.cpu())
    return float((expected.cpu()-actual).abs().max())

if __name__=="__main__":
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    result=attention_numeric()
    print("weights",result["weights"].tolist())
    print("context",result["context"].tolist())
    model=PatchEncoder()
    inputs=torch.randn(4,3,32,32)
    print("parameters",sum(p.numel() for p in model.parameters()))
    print("logit_shape",tuple(model(inputs).shape))
    difference=checkpoint_roundtrip(model,inputs,Path("checkpoints/patch_encoder.pt"))
    assert difference==0.
    print("checkpoint_max_difference",difference)
