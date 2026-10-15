"""Quick real-MFDiT CPU check; no MNIST download or training required."""
import torch
from models.dit import MFDiT
from experiment_core import training_loss,sample_one_step

def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model=MFDiT(input_size=32,patch_size=8,in_channels=1,dim=64,
                depth=1,num_heads=4,num_classes=10,max_cfg=4.0)
    x=torch.rand(3,1,32,32);labels=torch.tensor([1,2,3])
    for mode in ("mf","imf","imf_cfg"):
        model.zero_grad(set_to_none=True)
        loss,stats=training_loss(model,x,labels,mode)
        assert loss.isfinite(),f"{mode} non-finite loss"
        loss.backward()
        assert model.final_layer_u.linear.weight.grad is not None,mode
        assert torch.isfinite(model.final_layer_u.linear.weight.grad).all(),mode
        z,u=sample_one_step(model,torch.randn_like(x),labels,omega=2.)
        assert z.shape==x.shape and torch.isfinite(z).all()
        print(mode,"loss",round(float(loss.detach()),4),
              "u grad ok; 1-NFE output shape",tuple(z.shape))
    print("SMOKE TEST PASSED (not a quality or convergence benchmark)")

if __name__=="__main__":
    main()
