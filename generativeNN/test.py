from models import kl_divergence_gaussian
import torch

mean1 = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(4, 1)
log_std1 = torch.tensor([1.0, 1.0, 2.0, 2.0])
mean2 = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(4, 1)
log_std2 = torch.tensor([1.0, 1.0, 2.0, 2.0])

print(mean1.shape, log_std1.shape, mean2.shape, log_std2.shape)

kl = kl_divergence_gaussian(mean1, log_std1, mean2, log_std2)
print(kl)