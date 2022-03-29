from sys import prefix
from transformers import MT5ForConditionalGeneration
from torch.utils.data import DataLoader
from transformers import AdamW

import logging
import torch

from datasets import WebNLG
from models import PrefixTuning
from utils import generate_data

def main(n_epochs=5, lr=0.001, accum=32, preseqlen=5, hidden_dim=512):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = WebNLG(raw_path='data/release_v3.0/en/train', split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)

    test_dataset = WebNLG(raw_path='data/release_v3.0/en/dev', split='dev')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)

    # Load Pre-Trained Tokenizer, LM
    pretrained = MT5ForConditionalGeneration.from_pretrained("google/mt5-small" )
    pretrained = pretrained.to(device)
    pretrained.resize_token_embeddings(len(train_dataset.tokenizer))
    
    prefix_model = PrefixTuning(model=pretrained, preseqlen=preseqlen, hidden_dim=hidden_dim)
    prefix_model.to(device)

    optimizer = AdamW(prefix_model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):

        print('Running epoch: {}'.format(epoch))

        loss_train = 0

        for step, batch in enumerate(train_dataloader):

            print(f'{step}/{len(train_dataloader)}')

            prefix_model.train()

            samples = batch[0].squeeze().to(device)
            summaries = batch[1].squeeze().to(device)

            optimizer.zero_grad()

            # Get Past-Key-Values
            past_key_values = prefix_model(batch_size=samples.shape[0])
            prefix_model.past_key_values = past_key_values

            # Forward: Base (Pre-Trained) LM
            outputs = prefix_model.model(input_ids=samples, labels=summaries)

            loss = outputs[0] / accum
            loss.backward()
            loss_train += loss.item()
            
            if (step + 1) % accum == 0:

                # Set Loss to 0
                loss_train = 0

                optimizer.step()
                optimizer.zero_grad()

        generate_data(prefix_model, test_dataloader, test_dataset.tokenizer, device, epoch, lr, preseqlen, hidden_dim)

if __name__ == "__main__":
    main()