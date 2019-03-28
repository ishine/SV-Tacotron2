import torch
import SpeakerEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SpeakerEncoder.get_model().to(device)
test_input = torch.randn(2, 180, 80).to(device)
print(model(test_input).size())