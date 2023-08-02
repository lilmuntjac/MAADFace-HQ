from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import save_image

from src.tweaker import Tweaker, Losses
from src.utils import *

def main(args):
    print('Pytorch is running on version: ' + torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tweaker = Tweaker(batch_size=1, tweak_type=args.adv_type)
    # load the element
    element_path = Path(args.stats)
    adv_component = np.load(element_path)
    adv_component = torch.from_numpy(adv_component).to(device)
    print(f"The element is in shape of: {adv_component.shape}")
    # resolve the directory
    fig_root = Path(args.out_dir)
    fig_root.mkdir(parents=True, exist_ok=True)
    fig_path = fig_root / (args.name + '.png')

    # black_img = torch.zeros((args.batch_size, 3, 224, 224)).to(device)
    gray_img = torch.full((1, 3, 224, 224), 0.5).to(device)
    dummy_label = torch.zeros((1, 1)).to(device)

    match args.show_type:
        case "element":
            match args.adv_type:
                case "noise":
                    # enhance the noise to make it more obvious
                    adv_component*=8
                    element, _ = tweaker.apply(gray_img, dummy_label, adv_component)
                case "patch" | "frame" | "eyeglasses":
                    # save the raw image as-is
                    element = adv_component
        case "deploy":
            element, _ = tweaker.apply(gray_img, dummy_label, adv_component)
        case _:
            assert False, "unknown show type"
    
    save_image(element, fig_path)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="output the tweak element as image")
    # For input tweaking element
    parser.add_argument("--adv-type", default=None, type=str, help="type of adversarial element, only 'noise', 'patch', 'frame', and 'eyeglasses' are allowed")
    parser.add_argument("--stats", type=str, help="file path to the element .npy file")
    parser.add_argument("-o", "--out-dir", type=str, help="output folder for attributes confusion matrix")
    parser.add_argument("-n", "--name", type=str, help="output image name")

    parser.add_argument("--show-type", default="element" ,type=str, help="Get the element itself or the element apply onto the gray image. 'element' or 'deploy'")

    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)