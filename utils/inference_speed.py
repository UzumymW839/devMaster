import torch
import datetime

def inference_speed(model, test_tensor):
    '''Inference speed test for the model.'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    if test_tensor is None:
        test_tensor = torch.randn(1,5,32000).to(device)
    else:
        test_tensor = test_tensor.to(device)

    model.eval()
    with torch.no_grad():
        start_time = datetime.datetime.now()
        for i in range(100):
            model(test_tensor)
        end_time = datetime.datetime.now()
    model.train()
    
    return end_time - start_time