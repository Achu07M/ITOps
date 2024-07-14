import pandas as pd
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import joblib
import torch

def generate_synthetic_data():
    df_train = pd.read_csv('../ITAM_project/networkEvents.csv',sep=';',on_bad_lines='skip')
    sampled_name = random.choice(df_train['Name'].tolist())
    sampled_event = random.choice(df_train['Event'].tolist())
    synthetic_data = {
        'Name': sampled_name,
        'Event': sampled_event,
        'EventType': hex(random.randint(0, 2**32 - 1)),
        'CreatedOn': (datetime(2014, 9, 16, 1, 41, 49) + timedelta(seconds=random.randint(0, 86400))).strftime('%m/%d/%Y %H:%M:%S')
    }
    return synthetic_data

def load_model(model_dir):
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    le_severity = joblib.load(f"{model_dir}/le_severity.pkl")
    le_name = joblib.load(f"{model_dir}/le_name.pkl")
    le_event_type = joblib.load(f"{model_dir}/le_event_type.pkl")
    return model, tokenizer, le_severity, le_name, le_event_type
    
def predict_severity(model, tokenizer, event, max_len):
    encoding = tokenizer.encode_plus(
        event,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)

    return prediction.item()

def return_prediction(model_dir):
  model, tokenizer, le_severity,le_name, le_event_type = load_model(model_dir)
  max_len = 128
  synthetic_data = generate_synthetic_data()
  severity_prediction = predict_severity(model, tokenizer, synthetic_data['Event'], max_len)
  synthetic_data['Severity'] = le_severity.inverse_transform([severity_prediction])[0]
  return synthetic_data