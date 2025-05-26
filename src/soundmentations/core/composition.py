__all__ = [
    "BaseCompose",
    "Compose",
]

class BaseCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, samples, sample_rate=44100):
        for t in self.transforms:
            samples = t(samples, sample_rate)
        return samples
    
class Compose(BaseCompose):
    def __init__(self, transforms):
        super().__init__(transforms)