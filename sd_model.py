from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
import torch
import pickle as pkl


device = "cuda"

def get_cn_pipeline():
    controlnets = [
        ControlNetModel.from_pretrained("./controlnet/lineart", torch_dtype=torch.float16, use_safetensors=True),
        ControlNetModel.from_pretrained("mattyamonaca/controlnet_line2line_xl", torch_dtype=torch.float16)
    ]

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "cagliostrolab/animagine-xl-3.1", controlnet=controlnets, vae=vae, torch_dtype=torch.float16
    )

    return pipe

def get_ip_pipeline():
    controlnets = [
        ControlNetModel.from_pretrained("./controlnet/lineart", torch_dtype=torch.float16, use_safetensors=True),
        ControlNetModel.from_pretrained("mattyamonaca/controlnet_line2line_xl", torch_dtype=torch.float16)
    ]

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "cagliostrolab/animagine-xl-3.1", controlnet=controlnets, vae=vae, torch_dtype=torch.float16
    )
    pipe.load_ip_adapter(
        "ozzygt/sdxl-ip-adapter",
        "", 
        weight_name="ip-adapter_sdxl_vit-h.safetensors"
    )

    return pipe

def invert_image(img):
    # 画像を読み込む
    # 画像をグレースケールに変換（もしもともと白黒でない場合）
    img = img.convert('L')
    # 画像の各ピクセルを反転
    inverted_img = img.point(lambda p: 255 - p)
    # 反転した画像を保存
    return inverted_img


def get_cn_detector(image):
    re_image = invert_image(image)    
    detectors = [re_image, image]
    return detectors

