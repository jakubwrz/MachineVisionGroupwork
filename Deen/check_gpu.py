"""Verify the project's GPU training dependencies with real CUDA operations."""
import json
import sys
import torch
import torchvision
import ultralytics


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment')
    device = torch.device('cuda:0')
    a = torch.randn(512, 512, device=device, requires_grad=True)
    loss = (a @ a.T).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(a.grad).all()
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
    kept = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.5)
    assert kept.tolist() == [0]
    torch.cuda.synchronize()
    print(json.dumps(dict(python=sys.version, executable=sys.executable,
                         torch=torch.__version__, torchvision=torchvision.__version__,
                         ultralytics=ultralytics.__version__, cuda_runtime=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(0),
                         total_vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                         cuda_forward_backward='PASS', cuda_nms='PASS'), indent=2))


if __name__ == '__main__':
    main()
