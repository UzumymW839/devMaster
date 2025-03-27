import torch

def all_subclasses(cls):
    """
    Given a class cls, returns a list of all subclasses (including subclasses of subclasses).
    Subclasses here mean classes inheriting from a class.
    """
    return set([s for s in cls.__subclasses__()]).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])

def direct_subclasses(cls):
    """
    Given a class cls, return a list only of direct subclasses without subclasses of subclasses.
    """
    return [cls for cls in cls.__subclasses__() ]


class MockDataset(torch.utils.data.Dataset):
    """
    Mock dataset used in testing.
    """

    def __init__(self, full_length=100) -> None:
        super().__init__()
        self.datalist = list(range(100))

    def __len__(self) -> int:
        """
        Return the total number of images in the dataset.
        """
        return len(self.datalist)


def common_elements(list_a, list_b):
    return [fname for fname in list_a if fname in list_b]
