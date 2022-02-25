from sys import prefix
from transformers import GPT2LMHeadModel, GPT2Config
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

import logging
import torch

from datasets import WebNLG
from models import PrefixTuning

# logging.basicConfig(level=logging.ERROR)

def main(n_epochs=1, lr=0.001, accum=32):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = WebNLG()
    dataloader = DataLoader(dataset, batch_size=4)

    # Load Pre-Trained Tokenizer, LM
    pretrained = GPT2LMHeadModel.from_pretrained("gpt2")
    pretrained = pretrained.to(device)
    pretrained.resize_token_embeddings(len(dataset.tokenizer))

    config = GPT2Config()
    prefix_model = PrefixTuning(base_config=config)
    prefix_model.to(device)

    optimizer=AdamW(prefix_model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):

        print('Running epoch: {}'.format(epoch))

        loss_train=0

        for step, batch in enumerate(dataloader):

            print(step)

            prefix_model.train()

            samples = batch[0].squeeze().to(device)
            summaries = batch[1].squeeze().to(device)

            optimizer.zero_grad()

            # Get Past-Key-Values
            past_key_values = prefix_model(batch_size=samples.shape[0], device=device)

            # Forward: Base (Pre-Trained) LM
            outputs = pretrained(input_ids=samples, labels=summaries, past_key_values=past_key_values)

            loss = outputs[0] / accum
            loss.backward()
            loss_train += loss.item()
            
            if (step + 1) % accum == 0:

                # Set Loss to 0
                loss_train=0

                optimizer.step()
                optimizer.zero_grad()


if __name__ == "__main__":
    main()