import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm


# compute metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics

tax=pd.read_csv("Data/x_snc_taxonomy_sys_map_new.csv")
end_val=tax['ending_node'].to_list()
path=tax['path'].to_list()
path=list(map(lambda st:st.split('.'), path))
tax_path=dict(zip(end_val, path))
# print(tax_path)

# Confirm the data type of keys and values
print(f"Type of keys: {type(list(tax_path.keys())[0])}")
print(f"Type of values: {type(list(tax_path.values())[0])}")
# Check the number of entries in the dictionary
print(f"Number of entries in tax_path: {len(tax_path)}")

# Display keys and values
print("Sample keys:", list(tax_path.keys())[:5])  # Show the first 5 keys
print("Sample values:", list(tax_path.values())[:5])  # Show the first 5 values

# Select a specific key (replace 'key_value' with an actual key from your dataset)
key_value = list(tax_path.keys())[0]
print(f"Key: {key_value}")
print(f"Value: {tax_path[key_value]}")

# Analyze keys and values
key_lengths = [len(k) for k in tax_path.keys()]
value_lengths = [len(v) for v in tax_path.values()]

print(f"Average key length: {sum(key_lengths) / len(key_lengths)}")
print(f"Average value length: {sum(value_lengths) / len(value_lengths)}")


import matplotlib.pyplot as plt

# Plot distribution of value lengths
plt.hist(value_lengths, bins=20, edgecolor='k')
plt.title("Distribution of Path Lengths")
plt.xlabel("Path Length")
plt.ylabel("Frequency")
# plt.show()

import pandas as pd

# Convert dictionary to DataFrame for better visualization
tax_path_df = pd.DataFrame({'ending_node': list(tax_path.keys()), 'path': list(tax_path.values())})
print(tax_path_df.head())

kb_data=pd.read_parquet('Data/public_customer_kbs.parquet', engine='pyarrow')
print(kb_data.head())

kb_data = kb_data.rename(columns={'NUMBER': 'number', 'SHORT_DESCRIPTION': 'desc', 'TEXT':'text'})


kb_data=kb_data[['number','desc','text']]

allkbmeta1=pd.read_csv("Data/allkbs_meta1.csv")
allkbmeta2=pd.read_csv("Data/allkbs_meta2.csv")
allkbmeta= pd.concat([allkbmeta1,allkbmeta2])
taxresults=pd.read_csv("Data/allresults.csv") 
print(allkbmeta.head())

df2=pd.read_pickle('Data/taas_trainingdata500_3.pkl')
df=pd.read_pickle('Data/taas_trainingdata100.pkl')

df=df.reset_index(drop=True)
df = df.rename(columns={'label': 'og_label'})

df2=df2.reset_index(drop=True)
df2 = df2.rename(columns={'label': 'og_label'})

print(df2.head())
print(df.head())

# Display the first few rows
print("First few rows of the DataFrame:")
print(df.head())

# Display the column names and data types
print("\nColumn names and data types:")
print(df.dtypes)

# Check the shape of the DataFrame
print(f"\nNumber of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
# Display summary statistics for numerical columns
print("\nSummary statistics for numerical columns:")
print(df.describe())

# Display summary statistics for categorical columns
print("\nSummary statistics for categorical columns:")
print(df.describe(include=['object', 'category']))

# Check the number of unique values for each column
print("\nNumber of unique values in each column:")
print(df.nunique())

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['label'] = labelencoder.fit_transform(df['og_label'])
df2['label'] = labelencoder.fit_transform(df2['og_label'])



hold

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['label'] = labelencoder.fit_transform(df['og_label'])

df2['label'] = labelencoder.fit_transform(df2['og_label'])

len(set(df.label))

label_dict_100 = dict(zip(df.label,df.taxonomy))
label_dict_500 = dict(zip(df2.label,df2.taxonomy))
                      
len(label_dict_500)

label_dict_500 
len(label_dict_100) 

len(df.label)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
train_df, val_df = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
test_df, val_df = train_test_split(val_df, test_size=0.50, stratify=val_df['label'], random_state=42)

def encode(docs):
    '''
    This function takes list of texts and returns input_ids and attention_mask of texts
    '''
    encoded_dict = tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=512, padding='max_length',
                            return_attention_mask=True, truncation=True, return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks


train_input_ids, train_att_masks = encode(train_df['text'].values.tolist())
valid_input_ids, valid_att_masks = encode(val_df['text'].values.tolist())
test_input_ids, test_att_masks = encode(test_df['text'].values.tolist())

train_y = torch.LongTensor(train_df['label'].values.tolist())
valid_y = torch.LongTensor(val_df['label'].values.tolist())
test_y = torch.LongTensor(test_df['label'].values.tolist())
train_y.size(),valid_y.size(),test_y.size()

BATCH_SIZE = 32
train_dataset = TensorDataset(train_input_ids, train_att_masks, train_y)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

valid_dataset = TensorDataset(valid_input_ids, valid_att_masks, valid_y)
valid_sampler = SequentialSampler(valid_dataset)
valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE)

test_dataset = TensorDataset(test_input_ids, test_att_masks, test_y)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

N_labels = len(train_df.label.unique())
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=N_labels,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

model = model.to(device)

EPOCHS = 100
LEARNING_RATE = 2e-6

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, 
             num_warmup_steps=0,
            num_training_steps=len(train_dataloader)*EPOCHS )


import math

train_loss_per_epoch = []
val_loss_per_epoch = []
val_loss_max=float('inf')

for epoch_num in range(EPOCHS):
    print('Epoch: ', epoch_num + 1)
    '''
    Training
    '''
    model.train()
    train_loss = 0
    for step_num, batch_data in enumerate(tqdm(train_dataloader,desc='Training')):
        input_ids, att_mask, labels = [data.to(device) for data in batch_data]
        output = model(input_ids = input_ids, attention_mask=att_mask, labels= labels)
        
        loss = output.loss
        train_loss += loss.item()

        model.zero_grad()
        loss.backward()
        del loss

        clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    train_loss_per_epoch.append(train_loss / (step_num + 1))              


    '''
    Validation
    '''
    model.eval()
    valid_loss = 0
    valid_pred = []
    with torch.no_grad():
        for step_num_e, batch_data in enumerate(tqdm(valid_dataloader,desc='Validation')):
            input_ids, att_mask, labels = [data.to(device) for data in batch_data]
            output = model(input_ids = input_ids, attention_mask=att_mask, labels= labels)

            loss = output.loss
            valid_loss += loss.item()
   
            valid_pred.append(np.argmax(output.logits.cpu().detach().numpy(),axis=-1))
        
    val_loss_per_epoch.append(valid_loss / (step_num_e + 1))
    valid_pred = np.concatenate(valid_pred)
    if(valid_loss<val_loss_max):
        valid_loss_max=valid_loss
        torch.save(model.state_dict(), 'model.pk')
    else:
        break

    '''
    Loss message
    '''
    print("{0}/{1} train loss: {2} ".format(step_num+1, math.ceil(len(train_df) / BATCH_SIZE), train_loss / (step_num + 1)))
    print("{0}/{1} val loss: {2} ".format(step_num_e+1, math.ceil(len(val_df) / BATCH_SIZE), valid_loss / (step_num_e + 1)))
    print("Accuracy=",metrics.accuracy_score(val_df['label'].to_numpy(),valid_pred)
          

    model.eval()
    test_pred = []
    test_loss= 0
    with torch.no_grad():
        for step_num, batch_data in tqdm(enumerate(test_dataloader)):
            input_ids, att_mask, labels = [data.to(device) for data in batch_data]
            output = model(input_ids = input_ids, attention_mask=att_mask, labels= labels)

            loss = output.loss
            test_loss += loss.item()
    
            test_pred.append(np.argmax(output.logits.cpu().detach().numpy(),axis=-1))
    test_pred = np.concatenate(test_pred)
    )


    print('classifiation report')
    print(classification_report(test_pred, test_df['label'].to_numpy()))

