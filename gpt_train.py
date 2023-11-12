import torch
import gpt

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(gpt.model.parameters(), lr=gpt.learning_rate)
print("Start Training...")

for iter in range(gpt.max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % gpt.eval_interval == 0 or iter == gpt.max_iters - 1:
        losses = gpt.estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = gpt.get_batch('train')

    # evaluate the loss
    logits, loss = gpt.model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# After the training loop, save the model so you don't have to re-train every time:
torch.save(gpt.model.state_dict(), 'gpt_model.pth')
