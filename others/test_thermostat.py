import thermostat
import torch
from transformers import AutoModelForSequenceClassification

data = thermostat.load("imdb-albert-lig")

albert = AutoModelForSequenceClassification.from_pretrained(data.model_name, return_dict=False)
albert.eval()

sliced_data = data[:10]
thermostat_preds = [instance.predictions for instance in sliced_data]

batch_input = torch.tensor([instance.input_ids for instance in sliced_data])
preds = albert(batch_input)[0]

print("thermostat  ---  new inference".center(89, " "))
for m, n in zip(thermostat_preds, preds.tolist()):
    print(m, " --- ", n)
