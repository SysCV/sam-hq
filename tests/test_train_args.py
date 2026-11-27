import functools
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "train" / "train.py"

BASE_ARGV = [
    "train.py",
    "--checkpoint",
    "./pretrained_checkpoint/sam_vit_l_0b3195.pth",
    "--model-type",
    "vit_l",
    "--output",
    "work_dirs/hq_sam_l",
]


def _make_stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, val in attrs.items():
        setattr(mod, key, val)
    return mod


@functools.lru_cache(maxsize=1)
def _load_train_module() -> types.ModuleType:
    for stdlib_mod in ("argparse", "os", "random", "typing"):
        importlib.import_module(stdlib_mod)

    fn_mod = _make_stub_module("torch.nn.functional")
    nn_mod = _make_stub_module(
        "torch.nn",
        Module=type("Module", (), {}),
        functional=fn_mod,
    )
    torch_mod = _make_stub_module(
        "torch",
        Tensor=object,
        nn=nn_mod,
        optim=_make_stub_module("torch.optim"),
        autograd=_make_stub_module("torch.autograd", Variable=object()),
    )
    pyplot_mod = _make_stub_module("matplotlib.pyplot")
    modeling_mod = _make_stub_module(
        "segment_anything_training.modeling",
        TwoWayTransformer=type("TwoWayTransformer", (), {}),
        MaskDecoder=type("MaskDecoder", (), {}),
    )
    dataloader_mod = _make_stub_module(
        "utils.dataloader",
        get_im_gt_name_dict=lambda *a, **kw: None,
        create_dataloaders=lambda *a, **kw: None,
        RandomHFlip=object,
        Resize=object,
        LargeScaleJitter=object,
    )
    loss_mod = _make_stub_module(
        "utils.loss_mask",
        loss_masks=lambda *a, **kw: None,
    )
    misc_mod = _make_stub_module("utils.misc")
    stubs = {
        "numpy": _make_stub_module("numpy"),
        "torch": torch_mod,
        "torch.optim": torch_mod.optim,
        "torch.nn": nn_mod,
        "torch.nn.functional": fn_mod,
        "torch.autograd": torch_mod.autograd,
        "matplotlib": _make_stub_module("matplotlib", pyplot=pyplot_mod),
        "matplotlib.pyplot": pyplot_mod,
        "cv2": _make_stub_module("cv2"),
        "segment_anything_training": _make_stub_module(
            "segment_anything_training",
            sam_model_registry={},
            modeling=modeling_mod,
        ),
        "segment_anything_training.modeling": modeling_mod,
        "utils": _make_stub_module(
            "utils",
            dataloader=dataloader_mod,
            loss_mask=loss_mod,
            misc=misc_mod,
        ),
        "utils.dataloader": dataloader_mod,
        "utils.loss_mask": loss_mod,
        "utils.misc": misc_mod,
    }

    spec = importlib.util.spec_from_file_location("_sam_hq_train", TRAIN_PY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(mod)
    return mod


def test_get_args_parser_accepts_hyphenated_local_rank():
    train_mod = _load_train_module()
    with mock.patch.object(sys, "argv", [*BASE_ARGV, "--local-rank=5"]):
        args = train_mod.get_args_parser()
    assert args.local_rank == 5

    with mock.patch.object(sys, "argv", [*BASE_ARGV, "--local-rank", "2"]):
        args = train_mod.get_args_parser()
    assert args.local_rank == 2


def test_get_args_parser_accepts_underscored_local_rank():
    train_mod = _load_train_module()
    with mock.patch.object(sys, "argv", [*BASE_ARGV, "--local_rank", "3"]):
        args = train_mod.get_args_parser()
    assert args.local_rank == 3

    with mock.patch.object(sys, "argv", [*BASE_ARGV, "--local_rank=4"]):
        args = train_mod.get_args_parser()
    assert args.local_rank == 4


def test_get_args_parser_allows_omitted_local_rank():
    train_mod = _load_train_module()
    with mock.patch.object(sys, "argv", list(BASE_ARGV)):
        args = train_mod.get_args_parser()
    assert args.local_rank is None
