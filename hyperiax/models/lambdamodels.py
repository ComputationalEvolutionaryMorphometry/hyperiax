from .updownmodel import UpModel, UpDownModel, DownModel
from .updatemodel import UpdateModel

class UpLambda(UpModel):
    def __init__(self, up_fn, fuse_fn) -> None:
        super().__init__()

        self.up_fn = up_fn
        self.fuse_fn = fuse_fn

    def up(self, *args, **kwargs):
        return self.up_fn(*args, **kwargs)
    
    def fuse(self, *args, **kwargs):
        return self.fuse_fn(*args, **kwargs)
    
class UpDownLambda(UpDownModel):
    def __init__(self, up_fn, fuse_fn, down_fn) -> None:
        super().__init__()

        self.up_fn = up_fn
        self.fuse_fn = fuse_fn
        self.down_fn = down_fn

    def up(self, *args, **kwargs):
        return self.up_fn(*args, **kwargs)
    
    def fuse(self, *args, **kwargs):
        return self.fuse_fn(*args, **kwargs)
    
    def down(self, *args, **kwargs):
        return self.down_fn(*args, **kwargs)
    
class DownLambda(DownModel):
    def __init__(self, down_fn) -> None:
        super().__init__()

        self.down_fn = down_fn
    
    def down(self, *args, **kwargs):
        return self.down_fn(*args, **kwargs)
    
class UpdateLambda(UpdateModel):
    def __init__(self, update_fn) -> None:
        super().__init__()

        self.update_fn = update_fn

    def update(self, *args, **kwargs):
        return self.update_fn(*args, **kwargs)