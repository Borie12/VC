import torch
from torch.utils.data import DataLoader, Dataset
from collections.abc import Iterable
from typing import Optional, Callable, Dict, Any
from torchvision import transforms

class Prefetcher(Iterator):
    def __init__(self, loader: DataLoader):
        self._loader = loader
        self.loader = iter(self._loader)
        self.stream = torch.cuda.Stream()
        self.next_batch = next(self.loader)
        self.to_device(self.next_batch, "cuda")
        torch.cuda.synchronize()

    @staticmethod
    def to_device(batch: Any, device: str):
        if isinstance(batch, dict):
            for key in batch:
                batch[key] = batch[key].to(device)
        elif isinstance(batch, list):
            for i, item in enumerate(batch):
                batch[i] = item.to(device)
        else:
            batch = batch.to(device)
        return batch

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.loader = iter(self._loader)
            self.next_batch = next(self.loader)
        with torch.cuda.stream(self.stream):
            self.to_device(self.next_batch, "cuda")

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch


class BasicDataloader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[torch.utils.data.Sampler] = None,
        batch_sampler: Optional[torch.utils.data.Sampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        worker_init_fn: Optional[Callable] = None,
        prefetch: bool = False,
        batch_transforms: Optional[Callable] = None,
    ):
        if sampler is not None or batch_sampler is not None:
            shuffle = False
        if batch_sampler is not None:
            sampler = None
            batch_size = 1
            drop_last = False
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn if collate_fn is None else collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
        )

        self.batch_transforms = batch_transforms

        self.prefetch = prefetch
        if prefetch:
            self.iter_loader = Prefetcher(self.dataloader)
        else:
            self.iter_loader = iter(self.dataloader)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def to_device(batch: Any, device: str):
        if isinstance(batch, dict):
            for key in batch:
                batch[key] = batch[key].to(device)
        elif isinstance(batch, list):
            for i, item in enumerate(batch):
                batch[i] = item.to(device)
        else:
            batch = batch.to(device)
        return batch

    @torch.no_grad()
    def get_batch(self, device: str = "cuda") -> Optional[Dict[str, torch.Tensor]]:
        try:
            batch = next(self.iter_loader)
            self.to_device(batch, device)
            if self.batch_transforms is not None:
                batch = self.batch_transforms(batch)
        except StopIteration:
            batch = None
            self.iter_loader = iter(self.dataloader)
        return batch


# Advanced Data Augmentation
class AdvancedDataAugmentation:
    def __init__(self):
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if 'image' in batch:
            batch['image'] = self.augmentations(batch['image'])
        return batch


# Example usage:
if __name__ == "__main__":
    # Create a dataset (example)
    class ExampleDataset(Dataset):
        def __init__(self, transform=None):
            self.transform = transform

        def __len__(self):
            return 100  # Example length

        def __getitem__(self, idx):
            # Dummy image and label
            image = torch.rand(3, 224, 224)  # Example image tensor
            label = torch.tensor(0)  # Example label
            sample = {'image': image, 'label': label}
            if self.transform:
                sample = self.transform(sample)
            return sample

    # Create dataset and dataloader
    dataset = ExampleDataset(transform=AdvancedDataAugmentation())
    dataloader = BasicDataloader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        prefetch=True
    )

    # Fetch a batch
    batch = dataloader.get_batch()
    print(batch)
