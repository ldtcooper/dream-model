from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from dream_dataset import DreamsDataset
from utils import set_device, build_dream_data, format_time, save_train_data, get_current_dir, save_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
import random
import torch
import numpy as np
import time
from tqdm.auto import tqdm
import pandas as pd

print('Initializing Script')

device = set_device()

tokenizer = tokenizer = GPT2Tokenizer.from_pretrained(
    'gpt2-medium', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)
model.resize_token_embeddings(len(tokenizer))

seed_val = 662023
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

epochs = 10
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8

print('Merging Data')

dreams = build_dream_data()

train_dreams, test_dreams = train_test_split(dreams, test_size=0.2)
train_size = len(train_dreams)
test_size = len(test_dreams)
train_size, test_size

print('Creating Datasets')

train_dataset = DreamsDataset(train_dreams, tokenizer)
test_dataset = DreamsDataset(test_dreams, tokenizer)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1)

print('Preparing for Training')

# this produces sample output every 100 steps
sample_every = 100

optimizer = AdamW(model.parameters(),
                  lr=learning_rate,
                  eps=epsilon
                  )

total_steps = train_size * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps)

total_t0 = time.time()
training_stats = []

print('Starting Training')

for epoch in tqdm(range(epochs)):
    print(f'Epoch {epoch}')

    t0 = time.time()
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_loader):
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        model.zero_grad()
        output = model(b_input_ids, labels=b_labels,
                       attention_mask=b_masks, token_type_ids=None)

        loss = output[0]

        batch_loss = loss.item()
        total_train_loss += batch_loss

        if (step % sample_every == 0) and (step != 0):
            elapsed = format_time(time.time() - t0)
            # .format(step, len(train_dataloader), batch_loss, elapsed)
            print(
                f'Batch {batch} of {train_size} -- Loss: {batch_loss} -- Elapsed: {elapsed}.')

            model.eval()
            sample_outputs = model.generate(
                bos_token_id=random.randint(1, 30000),
                do_sample=True,
                top_k=50,
                max_length=200,
                top_p=0.95,
                num_return_sequences=1
            )

            for i, sample_output in enumerate(sample_outputs):
                print(
                    f"{i}: {tokenizer.decode(sample_output, skip_special_tokens=True)}")
            model.train()

        loss.backward()
        optimizer.step()
        scheduler.step()

        avg_train_loss = total_train_loss / train_size

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print(f"Average training loss: {avg_train_loss}")
        print(f"Training epoch took: {training_time}")

        # validation
        print(f'Validating Step {step}')
        t0 = time.time()
        model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in test_loader:
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, labels=b_labels,
                                attention_mask=b_masks, token_type_ids=None)
                loss = outputs[0]

            batch_loss = loss.item()
            total_eval_loss += batch_loss

    avg_val_loss = total_eval_loss / len(test_loader)
    validation_time = format_time(time.time() - t0)

    print(f"Validation Loss: {avg_val_loss}")
    print(f"Validation took: {validation_time}")

    training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("Training complete!")
print(f"Total training took {format_time(time.time()-total_t0)} (h:mm:ss)")

print('Saving Training Data')

train_stats_df = pd.DataFrame(training_stats)
train_stats_df.set_index('epoch')
save_train_data(train_stats_df)

print('Saving Models')

model_dir = get_current_dir() / 'models'
save_model(model_dir, model, tokenizer)

print('Done!')
