# Adjusted Rand Index (ARI) computation in PyTorch

Rand index adjusted for chance.

The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.

Shapes: [batch_size, nuber_of_groups, ... ] 
```python

# example for image
true_mask = torch.randint(0, 2, (10, 6, 1, 128, 128)).float()
pred_mask = torch.rand((10, 6, 1, 128, 128)).float()

ari = adjusted_rand_index(true_mask, pred_mask)
print(ari)
# tensor([-6.5167e-05, -6.5279e-05, -5.1253e-05,  9.4984e-06, -4.6251e-05,
#         -6.0256e-05, -6.4240e-05, -4.1156e-05, -3.8754e-05, -5.9850e-05])

```