from ..common import imread_rgb
from .base import Dataset


class BDD100K(Dataset):
    """RGB only, official 100k splits, 1280x720 cut into two 640x720 halves."""

    name = "bdd"
    sliced = True
    manual = ("https://bdd-data.berkeley.edu (registration): BDD100K '100K Images' "
              "(100k_images_{train,val,test}.zip), unzip into bdd/ so that bdd/images/100k/train exists")
    required = ("images/100k/train", "images/100k/val", "images/100k/test")

    def load(self, split, sample_id):
        return {"rgb": imread_rgb(self.root / "images/100k" / split / f"{sample_id}.jpg")}
