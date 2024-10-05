# train.py
import wandb
import random  # for demo script
import os

wandb_api_key = "522aca12c02f85859cdc9bd70e649cec09eaa182"
wandb.login(key=wandb_api_key, relogin=True)



epochs = 10
lr = 0.01

run = wandb.init(name = "jj")

offset = random.random() / 5
print(f"lr: {lr}")

# simulating a training run
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()