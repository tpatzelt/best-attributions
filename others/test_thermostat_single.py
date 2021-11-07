import thermostat
import torch
from transformers import AutoModelForSequenceClassification

data = thermostat.load("imdb-albert-lig")

albert = AutoModelForSequenceClassification.from_pretrained(data.model_name, return_dict=False)
albert.eval()

instance = data[0]
thermostat_preds = instance.predictions  # [-2.9755005836486816, 3.422632932662964]
input = torch.tensor([instance.input_ids])
input_without_padding = input[:, :179]

new_preds = albert(input)[0]  # [0.5287, 0.1149]
new_preds_without_padding = albert(input_without_padding)[0]  # [-2.9755,  3.4226]

assert not torch.all(torch.isclose(new_preds, torch.tensor([thermostat_preds])))
assert torch.all(torch.isclose(new_preds_without_padding, torch.tensor([thermostat_preds])))
