import os, torch, torch.nn as nn, torch.nn.functional as F
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for m in self.body:
            out = m(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        return out + base


class Uint8Wrapper(nn.Module):
    """uint8 NHWC [1,H,W,3] -> uint8 NHWC [1,4H,4W,3]. All pre/post baked in."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x_u8):
        x = x_u8.permute(0, 3, 1, 2).float() / 255.0   # NHWC u8 -> NCHW f32 0..1
        y = self.net(x)                                  # NCHW f32
        y = torch.clamp(y * 255.0, 0.0, 255.0).round()
        y = y.permute(0, 2, 3, 1).to(torch.uint8)        # -> NHWC u8
        return y


sd = torch.load('realesr-animevideov3.pth', map_location='cpu')
sd = sd.get('params', sd.get('params_ema', sd))
net = SRVGGNetCompact(); net.load_state_dict(sd, strict=True); net.eval()
model = Uint8Wrapper(net).eval()

dummy = torch.randint(0, 256, (1, 64, 96, 3), dtype=torch.uint8)
with torch.no_grad():
    out = model(dummy)
print('forward ok, in', tuple(dummy.shape), 'out', tuple(out.shape), 'dtype', out.dtype,
      'range', int(out.min()), int(out.max()))

torch.onnx.export(
    model, dummy, 'onnx/anime-x4-uint8.onnx',
    input_names=['image'], output_names=['upscaled'],
    dynamic_axes={'image': {1: 'h', 2: 'w'}, 'upscaled': {1: 'H', 2: 'W'}},
    opset_version=17, dynamo=False,
)
print('exported onnx/anime-x4-uint8.onnx', os.path.getsize('onnx/anime-x4-uint8.onnx'), 'bytes')
