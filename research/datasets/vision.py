import torch
import torchvision

# Define an easy way of fetching transforms
def get_transforms(transforms):
    assembled_transforms = []
    for transform_name, transform_kwargs in transforms:
        transform = vars(torchvision.transforms)[transform_name](**transform_kwargs)
        assembled_transforms.append(transform)
    return torchvision.transforms.Compose(assembled_transforms)

class TorchVisionDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, path, train, transforms=[("ToTensor", {}),], **kwargs):
        super().__init__()
        transforms = get_transforms(transforms)
        dataset_class = vars(torchvision.datasets)[dataset]
        self.dataset = dataset_class(path, train=train, transforms=transforms **kwargs)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
