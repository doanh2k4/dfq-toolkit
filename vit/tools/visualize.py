import torch
import torchvision


def denormalize(img):
    invTrans = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            torchvision.transforms.Normalize(
                mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
            ),
        ]
    )

    inv_tensor = invTrans(img)
    return torchvision.transforms.ToPILImage()(inv_tensor)


def showone(ckpt_path, normalize):
    ckpt = torch.load(ckpt_path)
    img = ckpt["img"].data[0]
    print(ckpt["img_metas"].data[0][0])
    if normalize:
        return denormalize(img)
    return torchvision.transforms.ToPILImage()(img)
