# %%
from crosscoders.utils import *
from crosscoders.crosscoder import CrossCoder
torch.set_grad_enabled(False)

cross_coder = CrossCoder.load("version_1", "1")

# %%

norms = cross_coder.W_dec.norm(dim=-1)
norms.shape

relative_norms = norms[:, 1] / norms.sum(dim=-1)
relative_norms.shape

fig = px.histogram(
    relative_norms.detach().cpu().numpy(), 
    title="Gemma 2 2B Base vs IT Model Diff",
    labels={"value": "Relative decoder norm strength"},
    nbins=200,
)

fig.update_layout(showlegend=False)
fig.update_yaxes(title_text="Number of Latents")

# Update x-axis ticks
fig.update_xaxes(
    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
    ticktext=['0', '0.25', '0.5', '0.75', '1.0']
)

fig.show()

# %%

import json

data = {
    "values" : relative_norms[relative_norms > 0.9].tolist(),
    "indices" : [i for i, v in enumerate(relative_norms) if v > 0.9]
}

with open("features.json", "w") as f:
    json.dump(data, f)

# %%
