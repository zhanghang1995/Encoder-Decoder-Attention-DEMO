from torch.utils import  data
class CustomDataset(data.Dataset):
    def __init__(self):
        #initial file path
        pass

    def __getitem__(self, index):
            # TODO
            # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
            # 2. Preprocess the data (e.g. torchvision.Transform).
            # 3. Return a data pair (e.g. image and label).
            # 这里需要注意的是，第一步：read one data，是一个data
        pass

    def __len__(self):
            # You should change 0 to the total size of your dataset.
        return 0
