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

df= pd.read_csv("Data/final_courses_with_paths2.csv" , encoding='ISO-8859-1')

# Load label dictionaries from pickle files
labels2 = pickle.load(open('Data/label_dict_100.pickle', 'rb'))
labels1 = pickle.load(open('Data/label_dict_500.pickle', 'rb'))



# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    """Load pre-trained models."""
    model1 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels1), output_attentions=False, output_hidden_states=False)
    model2 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels2), output_attentions=False, output_hidden_states=False)

    model1.load_state_dict(torch.load('Data/model_500.pk', map_location=torch.device('cpu')))
    model2.load_state_dict(torch.load('Data/model_100.pk', map_location=torch.device('cpu')))

    model1.to(device)
    model2.to(device)
    
    return model1, model2

def encode(docs):
    """Tokenize the input documents."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    encoded_dict = tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=512, padding='max_length',
                            return_attention_mask=True, truncation=True, return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

def get_test_data(list1):
    """Prepare the test DataLoader."""
    test_input_ids, test_att_masks = encode(list1)
    test_dataset = TensorDataset(test_input_ids, test_att_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)
    return test_dataloader

def bert_classify(test_dataloader, model):
    """Perform classification using BERT model."""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs, masks = batch
            outputs = model(inputs, attention_mask=masks)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    return preds

def bert_pred(test_data_list):
    """Predict using both models and return a DataFrame with predictions."""
    # Load models
    model1, model2 = load_models()
    
    # Get test DataLoader
    test_dataloader = get_test_data(test_data_list)
    
    # Get predictions from the first model
    preds1 = bert_classify(test_dataloader, model1)
    
    # Prepare DataFrame for original input data and initial predictions
    pred_df = pd.DataFrame({'short_description': test_data_list, 'pred_500': [labels1[i] for i in preds1]})
    
    # Filter out texts with prediction 0
    pred_0_df = pred_df[pred_df['pred_500'] == '0']
    
    if not pred_0_df.empty:
        # Get predictions from the second model
        pred_0_dataloader = get_test_data(pred_0_df['short_description'].tolist())
        preds2 = bert_classify(pred_0_dataloader, model2)
        
        # Update predictions for these texts
        pred_df.loc[pred_df['pred_500'] == '0', 'pred_100'] = [labels2[i] for i in preds2]
    
    # Apply ast.literal_eval to convert predictions strings into actual Python lists
    pred_df['pred_100'] = pred_df['pred_500'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    return pred_df

def predict_from_df(input_df):
    """Predict on the DataFrame and return predictions."""
    test_data_list = input_df['short_description'].values.tolist()
    predictions = bert_pred(test_data_list)
    return predictions

# Example usage:
# df = pd.read_csv('your_input_data.csv')  # Replace with your input file path
predictions_df = predict_from_df(df)
print(predictions_df)
print(predictions_df.head())
print(predictions_df.info())
predictions_df.to_csv('Data/results.csv')

