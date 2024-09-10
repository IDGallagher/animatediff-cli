import math

import matplotlib.pyplot as plt
import torch

max_len, d_model = 10, 8
position = torch.arange(max_len).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

plt.subplot(121)
plt.imshow(position.numpy())
plt.title('position')
plt.subplot(122)
plt.imshow(div_term.unsqueeze(0).numpy())
plt.title('div_term')
plt.show()
