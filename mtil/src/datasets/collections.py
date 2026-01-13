import os
import re
import pathlib
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from .cifar10 import CIFAR10 as cifar10, CIFAR100 as cifar100
from continuum.datasets import TinyImageNet200, ImageFolderDataset
import sys
from PIL import Image
def underline_to_space(s):
    return s.replace("_", " ")


class ClassificationDataset:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("./data"),
        batch_size=128,
        batch_size_eval=None,
        num_workers=16,
        append_dataset_name_to_template=False,
    ) -> None:
        self.name = "classification_dataset"
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        if batch_size_eval is None:
            self.batch_size_eval = batch_size
        else:
            self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers
        self.append_dataset_name_to_template = append_dataset_name_to_template

        self.train_dataset = self.test_dataset = None
        self.train_loader = self.test_loader = None
        self.classnames = None
        self.templates = None

    def build_dataloader(self):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def stats(self):
        L_train = len(self.train_dataset)
        L_test = len(self.test_dataset)
        N_class = len(self.classnames)
        return L_train, L_test, N_class

    @property
    def template(self):
        if self.append_dataset_name_to_template:
            return lambda x: self.templates[0](x)[:-1] + f", from dataset {self.name}]."
        return self.templates[0]

    def process_labels(self):
        self.classnames = [underline_to_space(x) for x in self.classnames]

    def split_dataset(self, dataset, ratio=0.8):
        train_size = int(ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        return train_dataset, test_dataset
    
    @property
    def class_to_idx(self):
        return {v: k for k, v in enumerate(self.classnames)}


class Aircraft(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "aircraft"
        self.train_dataset = datasets.FGVCAircraft(
            self.location, split="train", download=True, transform=self.preprocess
        )
        self.test_dataset = datasets.FGVCAircraft(
            self.location, split="test", download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of aircraft.",
            lambda c: f"a photo of the {c}, a type of aircraft.",
        ]

    # def process_labels(self):
    #     label = self.classnames
    #     for i in range(len(label)):
    #         if label[i].startswith("7"):
    #             label[i] = "Boeing " + label[i]
    #         elif label[i].startswith("An") or label[i].startswith("ATR"):
    #             pass
    #         elif label[i].startswith("A"):
    #             label[i] = "Airbus " + label[i]


class Caltech101(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "caltech101"
        dataset = datasets.Caltech101(
            self.location, download=True, transform=self.preprocess
        )
        self.classnames = dataset.categories

        train_dataset, test_dataset = self.split_dataset(dataset)
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.build_dataloader()

        self.classnames = [
            "off-center face",
            "centered face",
            "leopard",
            "motorbike",
            "accordion",
            "airplane",
            "anchor",
            "ant",
            "barrel",
            "bass",
            "beaver",
            "binocular",
            "bonsai",
            "brain",
            "brontosaurus",
            "buddha",
            "butterfly",
            "camera",
            "cannon",
            "side of a car",
            "ceiling fan",
            "cellphone",
            "chair",
            "chandelier",
            "body of a cougar cat",
            "face of a cougar cat",
            "crab",
            "crayfish",
            "crocodile",
            "head of a  crocodile",
            "cup",
            "dalmatian",
            "dollar bill",
            "dolphin",
            "dragonfly",
            "electric guitar",
            "elephant",
            "emu",
            "euphonium",
            "ewer",
            "ferry",
            "flamingo",
            "head of a flamingo",
            "garfield",
            "gerenuk",
            "gramophone",
            "grand piano",
            "hawksbill",
            "headphone",
            "hedgehog",
            "helicopter",
            "ibis",
            "inline skate",
            "joshua tree",
            "kangaroo",
            "ketch",
            "lamp",
            "laptop",
            "llama",
            "lobster",
            "lotus",
            "mandolin",
            "mayfly",
            "menorah",
            "metronome",
            "minaret",
            "nautilus",
            "octopus",
            "okapi",
            "pagoda",
            "panda",
            "pigeon",
            "pizza",
            "platypus",
            "pyramid",
            "revolver",
            "rhino",
            "rooster",
            "saxophone",
            "schooner",
            "scissors",
            "scorpion",
            "sea horse",
            "snoopy (cartoon beagle)",
            "soccer ball",
            "stapler",
            "starfish",
            "stegosaurus",
            "stop sign",
            "strawberry",
            "sunflower",
            "tick",
            "trilobite",
            "umbrella",
            "watch",
            "water lilly",
            "wheelchair",
            "wild cat",
            "windsor chair",
            "wrench",
            "yin and yang symbol",
        ]

        self.templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a painting of a {c}.",
            lambda c: f"a plastic {c}.",
            lambda c: f"a sculpture of a {c}.",
            lambda c: f"a sketch of a {c}.",
            lambda c: f"a tattoo of a {c}.",
            lambda c: f"a toy {c}.",
            lambda c: f"a rendition of a {c}.",
            lambda c: f"a embroidered {c}.",
            lambda c: f"a cartoon {c}.",
            lambda c: f"a {c} in a video game.",
            lambda c: f"a plushie {c}.",
            lambda c: f"a origami {c}.",
            lambda c: f"art of a {c}.",
            lambda c: f"graffiti of a {c}.",
            lambda c: f"a drawing of a {c}.",
            lambda c: f"a doodle of a {c}.",
            lambda c: f"a photo of the {c}.",
            lambda c: f"a painting of the {c}.",
            lambda c: f"the plastic {c}.",
            lambda c: f"a sculpture of the {c}.",
            lambda c: f"a sketch of the {c}.",
            lambda c: f"a tattoo of the {c}.",
            lambda c: f"the toy {c}.",
            lambda c: f"a rendition of the {c}.",
            lambda c: f"the embroidered {c}.",
            lambda c: f"the cartoon {c}.",
            lambda c: f"the {c} in a video game.",
            lambda c: f"the plushie {c}.",
            lambda c: f"the origami {c}.",
            lambda c: f"art of the {c}.",
            lambda c: f"graffiti of the {c}.",
            lambda c: f"a drawing of the {c}.",
            lambda c: f"a doodle of the {c}.",
        ]


class MNIST(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "mnist"
        self.train_dataset = datasets.MNIST(
            self.location, train=True, download=True, transform=self.preprocess
        )
        self.test_dataset = datasets.MNIST(
            self.location, train=False, download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f'a photo of the number: "{c}".',
        ]




class CIFAR10(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "cifar10"
        dataset = cifar10(preprocess=self.preprocess, location=self.location)

        self.train_dataset = dataset.train_dataset
        self.test_dataset = dataset.test_dataset
        self.build_dataloader()
        self.classnames = dataset.classnames
        self.process_labels()
        self.templates = dataset.template


class CIFAR100(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "cifar100"
        dataset = cifar100(preprocess=self.preprocess, location=self.location)

        self.train_dataset = dataset.train_dataset
        self.test_dataset = dataset.test_dataset
        self.build_dataloader()
        self.classnames = dataset.classnames
        self.process_labels()
        self.templates = dataset.template


class DTD(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "dtd"
        self.train_dataset = datasets.DTD(
            self.location, split="train", download=True, transform=self.preprocess
        )
        self.test_dataset = datasets.DTD(
            self.location, split="test", download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f'a photo of a {c} texture.',
            lambda c: f'a photo of a {c} pattern.',
            lambda c: f'a photo of a {c} thing.',
            lambda c: f'a photo of a {c} object.',
            lambda c: f'a photo of the {c} texture.',
            lambda c: f'a photo of the {c} pattern.',
            lambda c: f'a photo of the {c} thing.',
            lambda c: f'a photo of the {c} object.',
        ]


class EuroSAT(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "eurosat"
        dataset = datasets.EuroSAT(
            self.location, download=False, transform=self.preprocess
        )
        train_dataset, test_dataset = self.split_dataset(dataset)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.build_dataloader()

        self.classnames = [
            "annual crop land",
            "forest",
            "brushland or shrubland",
            "highway or road",
            "industrial buildings or commercial buildings",
            "pasture land",
            "permanent crop land",
            "residential buildings or homes or apartments",
            "river",
            "lake or sea",
        ]

        self.templates = [
            lambda c: f"a centered satellite photo of {c}.",
            lambda c: f"a centered satellite photo of a {c}.",
            lambda c: f"a centered satellite photo of the {c}.",
        ]

    def process_labels(self):
        super().process_labels()
        self.classnames = [re.sub(r"(\w)([A-Z])", r"\1 \2", x) for x in self.classnames]


class Flowers(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "flowers"
        self.train_dataset = datasets.Flowers102(
            self.location, split="train", download=True, transform=self.preprocess
        )
        self.test_dataset = datasets.Flowers102(
            self.location, split="test", download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = [
            "pink primrose",
            "hard-leaved pocket orchid",
            "canterbury bells",
            "sweet pea",
            "english marigold",
            "tiger lily",
            "moon orchid",
            "bird of paradise",
            "monkshood",
            "globe thistle",
            "snapdragon",
            "colt's foot",
            "king protea",
            "spear thistle",
            "yellow iris",
            "globe-flower",
            "purple coneflower",
            "peruvian lily",
            "balloon flower",
            "giant white arum lily",
            "fire lily",
            "pincushion flower",
            "fritillary",
            "red ginger",
            "grape hyacinth",
            "corn poppy",
            "prince of wales feathers",
            "stemless gentian",
            "artichoke",
            "sweet william",
            "carnation",
            "garden phlox",
            "love in the mist",
            "mexican aster",
            "alpine sea holly",
            "ruby-lipped cattleya",
            "cape flower",
            "great masterwort",
            "siam tulip",
            "lenten rose",
            "barbeton daisy",
            "daffodil",
            "sword lily",
            "poinsettia",
            "bolero deep blue",
            "wallflower",
            "marigold",
            "buttercup",
            "oxeye daisy",
            "common dandelion",
            "petunia",
            "wild pansy",
            "primula",
            "sunflower",
            "pelargonium",
            "bishop of llandaff",
            "gaura",
            "geranium",
            "orange dahlia",
            "pink-yellow dahlia",
            "cautleya spicata",
            "japanese anemone",
            "black-eyed susan",
            "silverbush",
            "californian poppy",
            "osteospermum",
            "spring crocus",
            "bearded iris",
            "windflower",
            "tree poppy",
            "gazania",
            "azalea",
            "water lily",
            "rose",
            "thorn apple",
            "morning glory",
            "passion flower",
            "lotus",
            "toad lily",
            "anthurium",
            "frangipani",
            "clematis",
            "hibiscus",
            "columbine",
            "desert-rose",
            "tree mallow",
            "magnolia",
            "cyclamen",
            "watercress",
            "canna lily",
            "hippeastrum",
            "bee balm",
            "ball moss",
            "foxglove",
            "bougainvillea",
            "camellia",
            "mallow",
            "mexican petunia",
            "bromelia",
            "blanket flower",
            "trumpet creeper",
            "blackberry lily",
        ]
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of flower.",
        ]


class Food(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "food"
        self.train_dataset = datasets.Food101(
            self.location, split="train", download=True, transform=self.preprocess
        )
        self.test_dataset = datasets.Food101(
            self.location, split="test", download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of food.",
        ]


class OxfordPet(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "oxford pet"
        self.train_dataset = datasets.OxfordIIITPet(
            self.location, split="trainval", download=True, transform=self.preprocess
        )
        self.test_dataset = datasets.OxfordIIITPet(
            self.location, split="test", download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of pet.",
        ]


class StanfordCars(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "stanford cars"
        
        # torchvision hardcodes "stanford_cars" (underscore), but we use "stanford-cars" (hyphen)
        # Replace the __init__ method to use the real directory name from the start
        original_init = datasets.StanfordCars.__init__
        import scipy.io as sio
        from torchvision.datasets.vision import VisionDataset
        from torchvision.datasets.utils import verify_str_arg
        from torchvision.datasets.folder import default_loader
        
        def patched_init(self_patched, root, split="train", **kwargs):
            try:
                import scipy.io as sio
                import numpy as np
            except ImportError:
                raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")
            
            # Call parent init
            VisionDataset.__init__(self_patched, root, transform=kwargs.get('transform'), 
                                  target_transform=kwargs.get('target_transform'))
            
            # Use stanford-cars (hyphen) instead of stanford_cars (underscore) - REAL DIRECTORY PATH
            self_patched._split = verify_str_arg(split, "split", ("train", "test"))
            self_patched._base_folder = pathlib.Path(root) / "stanford-cars"
            devkit = self_patched._base_folder / "devkit"
            
            if self_patched._split == "train":
                self_patched._annotations_mat_path = devkit / "cars_train_annos.mat"
                self_patched._images_base_path = self_patched._base_folder / "cars_train"
            else:
                # Test annotations should also be in devkit (we created cars_test_annos_withlabels.mat there)
                # But torchvision expects it in base folder, so check both locations
                test_anno_base = self_patched._base_folder / "cars_test_annos_withlabels.mat"
                test_anno_devkit = devkit / "cars_test_annos_withlabels.mat"
                if test_anno_base.exists():
                    self_patched._annotations_mat_path = test_anno_base
                elif test_anno_devkit.exists():
                    self_patched._annotations_mat_path = test_anno_devkit
                else:
                    self_patched._annotations_mat_path = test_anno_base  # Will fail with proper error
                self_patched._images_base_path = self_patched._base_folder / "cars_test"
            
            if kwargs.get('download', False):
                pass  # Skip download logic
            
            # Check existence
            if not devkit.is_dir():
                raise RuntimeError("Dataset not found.")
            if not (self_patched._annotations_mat_path.exists() and self_patched._images_base_path.is_dir()):
                raise RuntimeError("Dataset not found.")
            
            # Load samples and classes - only include files that actually exist
            # This handles cases where the dataset structure doesn't perfectly match annotations
            annotations = sio.loadmat(self_patched._annotations_mat_path, squeeze_me=True)["annotations"]
            self_patched._samples = []
            missing_count = 0
            for annotation in annotations:
                # Handle fname which might be a numpy array or string
                fname_raw = annotation["fname"]
                if isinstance(fname_raw, (np.ndarray, list)):
                    # Extract string from array/list
                    fname = str(fname_raw.item() if hasattr(fname_raw, 'item') else fname_raw[0])
                else:
                    fname = str(fname_raw)
                
                # Handle class which might be a numpy scalar
                class_raw = annotation["class"]
                class_idx = int(class_raw.item() if hasattr(class_raw, 'item') else class_raw) - 1
                
                img_path = self_patched._images_base_path / fname
                # Also try 6-digit version if 5-digit doesn't exist
                if not img_path.exists() and len(fname.split('.')[0]) == 5:
                    # Try 6-digit version
                    fname_6digit = '0' + fname
                    img_path_6digit = self_patched._images_base_path / fname_6digit
                    if img_path_6digit.exists():
                        img_path = img_path_6digit
                
                if img_path.exists():
                    self_patched._samples.append((str(img_path), class_idx))
                else:
                    missing_count += 1
            
            if missing_count > 0:
                print(f"  Warning: {missing_count} files from annotations not found, using {len(self_patched._samples)} available files")
            self_patched.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
            self_patched.class_to_idx = {cls: i for i, cls in enumerate(self_patched.classes)}
            self_patched.loader = kwargs.get('loader', default_loader)
        
        # Temporarily replace __init__
        datasets.StanfordCars.__init__ = patched_init
        
        try:
            self.train_dataset = datasets.StanfordCars(
                self.location, split="train", download=False, transform=self.preprocess
            )
            self.test_dataset = datasets.StanfordCars(
                self.location, split="test", download=False, transform=self.preprocess
            )
        finally:
            # Restore original __init__
            datasets.StanfordCars.__init__ = original_init
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of car.",
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
            lambda c: f"a photo of my {c}.",
            lambda c: f"i love my {c}!",
            lambda c: f"a photo of my dirty {c}.",
            lambda c: f"a photo of my clean {c}.",
            lambda c: f"a photo of my new {c}.",
            lambda c: f"a photo of my old {c}.",
        ]


class SUN397(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "sun397"
        # print('111')
        dataset = datasets.SUN397(
            self.location, download=False, transform=self.preprocess
        )

        train_dataset, test_dataset = self.split_dataset(dataset)
        # print('222')
        self.classnames = dataset.classes

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.build_dataloader()
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
        ]

class TinyImagenet(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "TinyImageNet"
        # Use real path: tinyimagenet/tiny-imagenet-200 instead of symlink
        # Check both possible locations (with and without tinyimagenet prefix)
        tinyimagenet_path = os.path.join(self.location, 'tiny-imagenet-200')
        tinyimagenet_real_path = os.path.join(self.location, 'tinyimagenet', 'tiny-imagenet-200')
        
        # Prefer real path, fallback to symlink location
        if os.path.exists(tinyimagenet_real_path) and os.path.isdir(tinyimagenet_real_path):
            dataset_path = tinyimagenet_real_path
        elif os.path.exists(tinyimagenet_path) and os.path.isdir(tinyimagenet_path):
            dataset_path = tinyimagenet_path
        else:
            # Default to real path (will error if not found, which is expected)
            dataset_path = tinyimagenet_real_path
        
        self.train_dataset = TinyImageNet_dataset(
            dataset_path, train=True, transform=self.preprocess
        )
        self.test_dataset = TinyImageNet_dataset(
            dataset_path, train=False, transform=self.preprocess
        )
        self.build_dataloader()


class TinyImageNet_dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")
        # self.dataset_train = TinyImageNet(root, train=True, transform=None)
        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def build_dataloader(self):
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=16,
            shuffle=True,
            num_workers=1,
        )


    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt