def expand_to_batchsize_for_layer(tup, batch_size, layer_id):
    return tup[layer_id].expand(-1, batch_size,-1,-1,-1)

def generate_data(prefix_model, dataloader, tokenizer, device):
    for step, batch in enumerate(dataloader):
        prefix_model.eval()

        samples = batch[0].squeeze().to(device)

        # Get Past-Key-Values
        past_key_values = prefix_model(batch_size=samples.shape[0])
        prefix_model.past_key_values = past_key_values

        outputs = prefix_model.model.generate(samples)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        