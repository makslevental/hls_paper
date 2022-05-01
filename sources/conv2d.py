class Mod(torch.nn.Module):
   def __init__(self):
       super(Mod, self).__init__()
       self.conv = torch.nn.Conv2d(in_channel=1, out_channels=4, 
          kernel_size=3, padding=3)

   def forward(self, x):
       return self.conv(x)