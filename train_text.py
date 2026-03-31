import json

from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import GPTNeoForSequenceClassification
import torch
import os
import logging
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from modules.dataset import HintsOfTruthTextDataset, TextDataset
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# CONFIG
# Define the path to your config file
config_path = 'configs/multimodal_mem_config.json'

# Open and read the JSON file
with open(config_path, 'r') as file:
    config = json.load(file)

model_name = 'bert'  # bert, gpt
MAX_LENGTH = 512

dataset = config.get('dataset', 'food_review')
train_file = config.get('train_file', '')
val_file = config.get('val_file', '')
logging_file = config.get('logging_file', '')
output_dir = config.get('output_dir', '')
# Training hyperparameters
EPOCHS = config.get('EPOCHS', 100)
BATCH_SIZE = config.get('BATCH_SIZE', 16)
LR = config.get('LR', 0.0001)
EARLY_STOP = config.get('EARLY_STOP', 10)

# LR decay config
LR_STEP_SIZE = config.get('LR_STEP_SIZE', 5)   # decay every 5 epochs
LR_GAMMA = config.get('LR_GAMMA', 0.5)         # multiply lr by 0.5

# wandb config
use_wandb = config.get('use_wandb', True)
wandb_project = config.get(
    'wandb_project', 'Multimodal synthesis data detection')
wandb_run_name = config.get('wandb_run_name', f'{model_name}-{dataset}-epo')
wandb_mode = config.get('wandb_mode', 'online')  # online, offline, disabled
if config.get("api_key"):
    os.environ["WANDB_API_KEY"] = config["api_key"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
available_models = ['bert', 'gpt']
best_acc = 0

# MODEL SELECTION
if model_name == 'bert':
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
elif model_name == 'gpt':
    model = GPTNeoForSequenceClassification.from_pretrained(
        "EleutherAI/gpt-neo-125M")
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
else:
    pass
model = model.to(device)
print(f'Model {model_name} loaded.')

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# Initialize wandb
if use_wandb and WANDB_AVAILABLE:
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        mode=wandb_mode,
        config=config
    )
    print('wandb initialized.')
elif use_wandb and not WANDB_AVAILABLE:
    print('wandb is not installed. Continuing without wandb.')


# DATA
if dataset == "hints_of_truth":
    train = HintsOfTruthTextDataset(
        train_file, "dev1", tokenizer, MAX_LENGTH)
    val = HintsOfTruthTextDataset(
        val_file, "dev2", tokenizer, MAX_LENGTH)
else:
    train = TextDataset(train_file, tokenizer, MAX_LENGTH)
    val = TextDataset(val_file, tokenizer, MAX_LENGTH)

train_dataloader = DataLoader(train, BATCH_SIZE, shuffle=True)
print(f'Loaded Traininig File: {train_file}.')
val_dataloader = DataLoader(val, BATCH_SIZE, shuffle=False)
print(f'Loaded Validation File: {val_file}.')
print('Data loaded.')

# logging
logging.basicConfig(filename=logging_file, level=logging.INFO, filemode='a+')
print('Log file initialized.')

# OPTIMIZER
optimiser = AdamW(model.parameters(), lr=LR)
scheduler = StepLR(optimiser, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

# OPTIMIZATION
print('Training..')
count = 0
for epoch in range(1, EPOCHS):
    model.train()

    pred_val = []
    labels_val = []
    train_loss = 0.0

    print(f'Epoch: {epoch}')
    for i, batch in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        optimiser.zero_grad()
        if i % 100 == 0:
            print(f'{i}th batch..')

        inputs = batch['input'].to(device)
        input_ids = inputs['input_ids'].squeeze(1)
        attention_mask = inputs['attention_mask'].squeeze(1)
        labels = torch.tensor(batch['label'])
        labels = labels.to(device)

        # output = model(**inputs, labels=labels)
        output = model(input_ids=input_ids,
                       attention_mask=attention_mask, labels=labels)
        loss = output.loss

        loss.backward()
        optimiser.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dataloader)

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        print('Validating..')
        for j, batchv in enumerate(val_dataloader):
            inputs_val = batchv['input'].to(device)
            input_ids_val = inputs_val['input_ids'].squeeze(1)
            attention_mask_val = inputs_val['attention_mask'].squeeze(1)
            label_val = batchv['label'].numpy().tolist()

            output_val = model(input_ids=input_ids_val,
                               attention_mask=attention_mask_val)

            loss_val = output.loss
            val_loss += loss_val.item()

            output_val = torch.softmax(output_val.logits, dim=-1)
            predictions = torch.argmax(
                output_val, dim=-1).detach().cpu().numpy().tolist()

            pred_val.extend(predictions)
            labels_val.extend(label_val)

        avg_val_loss = val_loss / len(val_dataloader)
        acc = accuracy_score(pred_val, labels_val)
        prec = precision_score(pred_val, labels_val)
        rec = recall_score(pred_val, labels_val)
        f1 = f1_score(pred_val, labels_val)

        logging.info(
            f'Epoch: {epoch}, Accuracy: {acc}, LR: {LR}, Batch Size: {BATCH_SIZE}.')
        print(f'# Train Loss: {avg_train_loss}')
        print(f'# Val Loss: {avg_val_loss}')
        print(f'# Accuracy: {acc}')
        print(f'# Precision: {prec}')
        print(f'# Recall: {rec}')
        print(f'# F1-score: {f1}')
        print(f'# Accuracy: {acc}')

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "val/accuracy": acc,
                "val/precision": prec,
                "val/recall": rec,
                "val/f1_score": f1,
                "lr": optimiser.param_groups[0]['lr']
            })

        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(output_dir, f'weight-{epoch}')
            model.save_pretrained(save_path)
            print('Saved model.')

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "best_val_accuracy": best_acc,
                    "best_epoch": epoch
                })
                wandb.save(save_path)

            count = 0
        else:
            count += 1

        if count == 5:
            print(f'Stopping at epoch: {epoch}')
            break
    # step lr decay after each epoch
    scheduler.step()

    print()

if use_wandb and WANDB_AVAILABLE:
    wandb.finish()
