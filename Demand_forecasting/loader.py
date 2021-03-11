import torch
import numpy as np
from sequences import *



def get_loader(data_dir, batch_size=25):
    dataset = forecast(data_dir)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False
                                              )
    return data_loader




data_dir = "C:\\Users\\arsal\\Downloads\\Demand_forecasting\\data"

loader=get_loader(data_dir)
dir(loader)

dd=[]
for batch in enumerate(loader):
    print(batch)
    dd.append(batch)