import numpy as np
import pandas as pd
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
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
import ast
import pickle  # I

# Sample DataFrame (for illustration purposes)
# df = pd.DataFrame({
#     'text': ["This is a positive statement.", "This is a negative statement."],
#     'label': [1, 0]
# })
df= pd.read_csv("Data/final_courses_with_paths2.csv" , encoding='ISO-8859-1', nrows=30)

# Load label dictionaries from pickle files
labels2 = pickle.load(open('Data/label_dict_100.pickle', 'rb'))
labels1 = pickle.load(open('Data/label_dict_500.pickle', 'rb'))


import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
import ast

# Function to load the pre-trained BERT models
# Function to load the pre-trained BERT models
def load_models():
    """
    Load the pre-trained BERT models from saved files.
    
    Returns:
        model1 (BertForSequenceClassification): The model for label '500'.
        model2 (BertForSequenceClassification): The model for label '100'.
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model for label '500'
    model1 = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=len(labels1),
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    model1.load_state_dict(torch.load('Data/model_500.pk', map_location=torch.device('cpu')))
    model1 = model1.to(device)

    # Load the model for label '100'
    model2 = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=len(labels2),
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    model2.load_state_dict(torch.load('Data/model_100.pk', map_location=torch.device('cpu')))
    model2 = model2.to(device)

    return model1, model2

# Function to encode texts for BERT input
def encode(docs):
    """
    This function takes a list of texts and returns input_ids and attention_mask.
    
    Args:
        docs (list): List of text data to be encoded.
        
    Returns:
        input_ids (torch.Tensor): Encoded input ids.
        attention_masks (torch.Tensor): Attention masks.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    encoded_dict = tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=512, padding='max_length',
                                               return_attention_mask=True, truncation=True, return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

# Function to prepare the data loader for prediction
def get_test_data(list1):
    """
    Prepare the DataLoader for the test data.
    
    Args:
        list1 (list): List of text data for testing.
        
    Returns:
        DataLoader: DataLoader object for the test data.
    """
    test_input_ids, test_att_masks = encode(list1)
    test_dataset = TensorDataset(test_input_ids, test_att_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)
    return test_dataloader

# Function to classify using BERT model
def bert_classify(test_dataloader, model):
    """
    Run the BERT model on the test data to get predictions.
    
    Args:
        test_dataloader (DataLoader): DataLoader object for the test data.
        model (BertForSequenceClassification): The BERT model for prediction.
        
    Returns:
        list: Predictions from the model.
    """
    model.eval()
    all_preds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)  # Move the batch to the appropriate device
            input_ids, attention_mask = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
    return all_preds

# Function to predict using both models
def bert_pred(test_data):
    """
    Predict using both BERT models for different scenarios.
    
    Args:
        test_data (list): List of text data for prediction.
        
    Returns:
        list: Predictions after applying both models.
    """
    # Load the models
    model1, model2 = load_models()

    # Prepare the test data
    test_dataloader = get_test_data(test_data)
    
    # Get predictions using the first model
    preds = bert_classify(test_dataloader, model1)
    
    # Filter the data that was classified as 0
    pred_0_df = [test_data[i] for i in range(len(preds)) if preds[i] == 0]
    
    if len(pred_0_df) != 0:
        # Prepare the DataLoader for filtered data
        pred_0_dataset_loader = get_test_data(pred_0_df)
        
        # Get predictions using the second modelz
        preds_2 = bert_classify(pred_0_dataset_loader, model2)
        
        preds_new = []
        j = 0
        for i in range(len(preds)):
            if preds[i] == 0:
                preds_new.append(ast.literal_eval(labels2[preds_2[j]]))
                j += 1
            else:
                preds_new.append(ast.literal_eval(labels1[preds[i]]))
        return preds_new
    
    # If no filtered data is there, just return predictions from the first model
    preds_new = [ast.literal_eval(labels1[i]) for i in preds]
    return preds_new

# Example usage
if __name__ == "__main__":
    # Assuming 'no_breadcrumb_df' is a DataFrame containing the input data
    test_data_list = df['short_description'].values.tolist()
    predictions = bert_pred(test_data_list)

    print(predictions)