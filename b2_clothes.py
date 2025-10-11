import torch
import comfy.utils
from comfy.model_management import processing_interrupted
from .support import (
    log, any_typ, get_device_list,
    pil2tensor, tensor2pil, expand_mask_batch,
    load_segmentation_model, load_VITMatte_model, 
    get_segmentation_batch, histogram_remap_batch,
    generate_VITMatte_trimap_batch, generate_VITMatte_batch,
    guided_filter_alpha_batch, mask_edge_detail_batch
)

device_list = get_device_list()

class Segformer_B2_Clothes_Fast:
    @classmethod
    def INPUT_TYPES(cls):
        method_list = ['VITMatte_fast', 'VITMatte', 'GuidedFilter', 'PyMatting', ]
        model_list = ['mattmdjaga/segformer_b2_clothes', 'sayeed99/segformer-b2-fashion', ]
        return {"required":
            {
                "image": ("IMAGE",),
                "labels": (any_typ),
                "model": (model_list,),
                "batch_size": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
                "detail_erode": ("INT", {"default": 12, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "detail_method": (method_list, {"default": "VITMatte_fast"}),
                "expand_mask": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "black_point": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "device": (device_list, {"default": device_list[0]}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "segformer_ultra_fast"
    CATEGORY = 'Mask'

    def segformer_ultra_fast(self, image, labels, model, batch_size, max_megapixels, detail_erode, detail_dilate,
        process_detail, detail_method, expand_mask, tapered_corners, black_point, white_point, device):
        
        total_images = image.shape[0]
        pbar = comfy.utils.ProgressBar(total_images)
        
        processed_image_chunks = []
        processed_mask_chunks = []

        model, processor = load_segmentation_model(device=device)

        for i in range(0, total_images, batch_size):
            if processing_interrupted():
                return (None, None)

            image_chunk = image[i:i + batch_size]
            image_chunk_on_device = image_chunk.to(device)
            pred_seg_chunk = get_segmentation_batch(image_chunk_on_device, model, processor, device)

            labels_to_keep = labels
            labels_tensor = torch.tensor(labels_to_keep, device=device).view(1, -1, 1, 1)
            mask_chunk_uint8 = torch.any(pred_seg_chunk.unsqueeze(1) == labels_tensor, dim=1)
            final_mask_chunk = (1 - mask_chunk_uint8.float()).unsqueeze(1)

            if process_detail:
                if detail_method == 'GuidedFilter':
                    processed_mask = guided_filter_alpha_batch(image_chunk_on_device, final_mask_chunk, detail_erode + detail_dilate // 6 + 1)
                    final_mask_chunk = histogram_remap_batch(processed_mask, black_point, white_point)
                elif detail_method == 'PyMatting':
                    detail_range = detail_erode + detail_dilate
                    processed_mask = mask_edge_detail_batch(image_chunk_on_device, final_mask_chunk, detail_range // 8 + 1, black_point, white_point)
                    final_mask_chunk = processed_mask
                else:
                    fast = "fast" in detail_method
                    trimap_chunk = generate_VITMatte_trimap_batch(final_mask_chunk, detail_erode, detail_dilate)
                    matte_chunk = generate_VITMatte_batch(image_chunk_on_device, trimap_chunk, device, max_megapixels, fast_mode=fast)
                    final_mask_chunk = matte_chunk
                    final_mask_chunk = histogram_remap_batch(final_mask_chunk, black_point, white_point)
            
            if expand_mask != 0:
                final_mask_chunk = expand_mask_batch(final_mask_chunk, expand_mask, tapered_corners)
            
            image_chunk_bchw = image_chunk_on_device.permute(0, 3, 1, 2)
            final_images_chunk_bchw = torch.cat((image_chunk_bchw, final_mask_chunk), dim=1)
            final_images_chunk = final_images_chunk_bchw.permute(0, 2, 3, 1).contiguous()
            final_masks_chunk = final_mask_chunk.squeeze(1)
        
            processed_image_chunks.append(final_images_chunk.cpu())
            processed_mask_chunks.append(final_masks_chunk.cpu())
            pbar.update(image_chunk.shape[0])

        final_images = torch.cat(processed_image_chunks, dim=0)
        final_masks = torch.cat(processed_mask_chunks, dim=0)
        
        return (final_images, final_masks)

class Segformer_B2_Clothes_Labels:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "face_torso": ("BOOLEAN", {"default": False}), "hair": ("BOOLEAN", {"default": False}),
                "hat": ("BOOLEAN", {"default": False}), "sunglass": ("BOOLEAN", {"default": False}),
                "left_arm": ("BOOLEAN", {"default": False}), "right_arm": ("BOOLEAN", {"default": False}),
                "left_leg": ("BOOLEAN", {"default": False}), "right_leg": ("BOOLEAN", {"default": False}),
                "upper_clothes": ("BOOLEAN", {"default": False}), "skirt": ("BOOLEAN", {"default": False}),
                "pants": ("BOOLEAN", {"default": False}), "dress": ("BOOLEAN", {"default": False}),
                "belt": ("BOOLEAN", {"default": False}), "shoe": ("BOOLEAN", {"default": False}),
                "bag": ("BOOLEAN", {"default": False}), "scarf": ("BOOLEAN", {"default": False}),
                "everything_else": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = (any_typ, )
    RETURN_NAMES = ("labels",)
    FUNCTION = "get_labels"
    CATEGORY = 'Mask'
    
    def get_labels(self, face_torso, hat, hair, sunglass, upper_clothes, skirt, pants, dress, belt,
        shoe, left_leg, right_leg, left_arm, right_arm, bag, scarf, everything_else):
        
        labels_to_keep = []
        if not everything_else: labels_to_keep.append(0)
        if not hat: labels_to_keep.append(1)
        if not hair: labels_to_keep.append(2)
        if not sunglass: labels_to_keep.append(3)
        if not upper_clothes: labels_to_keep.append(4)
        if not skirt: labels_to_keep.append(5)
        if not pants: labels_to_keep.append(6)
        if not dress: labels_to_keep.append(7)
        if not belt: labels_to_keep.append(8)
        if not shoe: labels_to_keep.extend([9, 10])
        if not face_torso: labels_to_keep.append(11)
        if not left_leg: labels_to_keep.append(12)
        if not right_leg: labels_to_keep.append(13)
        if not left_arm: labels_to_keep.append(14)
        if not right_arm: labels_to_keep.append(15)
        if not bag: labels_to_keep.append(16)
        if not scarf: labels_to_keep.append(17)
            
        return (labels_to_keep,)
    
class Segformer_B2_Fashion_Labels:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "shirt_blouse": ("BOOLEAN", {"default": False}), "top_tShirt_sweatshirt": ("BOOLEAN", {"default": False}),
                "sweater": ("BOOLEAN", {"default": False}), "cardigan": ("BOOLEAN", {"default": False}),
                "jacket": ("BOOLEAN", {"default": False}), "vest": ("BOOLEAN", {"default": False}),
                "pants": ("BOOLEAN", {"default": False}), "shorts": ("BOOLEAN", {"default": False}),
                "skirt": ("BOOLEAN", {"default": False}), "coat": ("BOOLEAN", {"default": False}),
                "dress": ("BOOLEAN", {"default": False}), "jumpsuit": ("BOOLEAN", {"default": False}),
                "cape": ("BOOLEAN", {"default": False}), "glasses": ("BOOLEAN", {"default": False}),
                "hat": ("BOOLEAN", {"default": False}), "headband_hairAccessory": ("BOOLEAN", {"default": False}),
                "tie": ("BOOLEAN", {"default": False}), "glove": ("BOOLEAN", {"default": False}),
                "watch": ("BOOLEAN", {"default": False}), "belt": ("BOOLEAN", {"default": False}),
                "leg_warmer": ("BOOLEAN", {"default": False}), "tights_stockings": ("BOOLEAN", {"default": False}),
                "sock": ("BOOLEAN", {"default": False}), "shoe": ("BOOLEAN", {"default": False}),
                "bag_wallet": ("BOOLEAN", {"default": False}), "scarf": ("BOOLEAN", {"default": False}),
                "umbrella": ("BOOLEAN", {"default": False}), "hood": ("BOOLEAN", {"default": False}),
                "collar": ("BOOLEAN", {"default": False}), "lapel": ("BOOLEAN", {"default": False}),
                "epaulette": ("BOOLEAN", {"default": False}), "sleeve": ("BOOLEAN", {"default": False}),
                "pocket": ("BOOLEAN", {"default": False}), "neckline": ("BOOLEAN", {"default": False}),
                "buckle": ("BOOLEAN", {"default": False}), "zipper": ("BOOLEAN", {"default": False}),
                "applique": ("BOOLEAN", {"default": False}), "bead": ("BOOLEAN", {"default": False}),
                "bow": ("BOOLEAN", {"default": False}), "flower": ("BOOLEAN", {"default": False}),
                "fringe": ("BOOLEAN", {"default": False}), "ribbon": ("BOOLEAN", {"default": False}),
                "rivet": ("BOOLEAN", {"default": False}), "ruffle": ("BOOLEAN", {"default": False}),
                "sequin": ("BOOLEAN", {"default": False}), "tassel": ("BOOLEAN", {"default": False}),
                "everything_else": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = (any_typ, )
    RETURN_NAMES = ("labels",)
    FUNCTION = "get_labels"
    CATEGORY = 'Mask'
    
    def get_labels(self, shirt_blouse, top_tShirt_sweatshirt, sweater, cardigan, jacket, vest,
        pants, shorts, skirt, coat, dress, jumpsuit, cape, glasses, hat, headband_hairAccessory,
        tie, glove, watch, belt, leg_warmer, tights_stockings, sock, shoe, bag_wallet, scarf,
        umbrella, hood, collar, lapel, epaulette, sleeve, pocket, neckline, buckle, zipper,
        applique, bead, bow, flower, fringe, ribbon, rivet, ruffle, sequin, tassel, everything_else):
        
        labels_to_keep = []
        if not everything_else: labels_to_keep.append(0)
        if not shirt_blouse: labels_to_keep.append(1)
        if not top_tShirt_sweatshirt: labels_to_keep.append(2)
        if not sweater: labels_to_keep.append(3)
        if not cardigan: labels_to_keep.append(4)
        if not jacket: labels_to_keep.append(5)
        if not vest: labels_to_keep.append(6)
        if not pants: labels_to_keep.append(7)
        if not shorts: labels_to_keep.append(8)
        if not skirt: labels_to_keep.append(9)
        if not coat: labels_to_keep.append(10)
        if not dress: labels_to_keep.append(11)
        if not jumpsuit: labels_to_keep.append(12)
        if not cape: labels_to_keep.append(13)
        if not glasses: labels_to_keep.append(14)
        if not hat: labels_to_keep.append(15)
        if not headband_hairAccessory: labels_to_keep.append(16)
        if not tie: labels_to_keep.append(17)
        if not glove: labels_to_keep.append(18)
        if not watch: labels_to_keep.append(19)
        if not belt: labels_to_keep.append(20)
        if not leg_warmer: labels_to_keep.append(21)
        if not tights_stockings: labels_to_keep.append(22)
        if not sock: labels_to_keep.append(23)
        if not shoe: labels_to_keep.append(24)
        if not bag_wallet: labels_to_keep.append(25)
        if not scarf: labels_to_keep.append(26)
        if not umbrella: labels_to_keep.append(27)
        if not hood: labels_to_keep.append(28)
        if not collar: labels_to_keep.append(29)
        if not lapel: labels_to_keep.append(30)
        if not epaulette: labels_to_keep.append(31)
        if not sleeve: labels_to_keep.append(32)
        if not pocket: labels_to_keep.append(33)
        if not neckline: labels_to_keep.append(34)
        if not buckle: labels_to_keep.append(35)
        if not zipper: labels_to_keep.append(36)
        if not applique: labels_to_keep.append(37)
        if not bead: labels_to_keep.append(38)
        if not bow: labels_to_keep.append(39)
        if not flower: labels_to_keep.append(40)
        if not fringe: labels_to_keep.append(41)
        if not ribbon: labels_to_keep.append(42)
        if not rivet: labels_to_keep.append(43)
        if not ruffle: labels_to_keep.append(44)
        if not sequin: labels_to_keep.append(45)
        if not tassel: labels_to_keep.append(46)
            
        return (labels_to_keep,)

NODE_CLASS_MAPPINGS = {
    "SegformerB2ClothesUltraBatch": Segformer_B2_Clothes_Fast,
    "Segformer_B2_Clothes_Labels": Segformer_B2_Clothes_Labels,
    "Segformer_B2_Clothes_Fashion_Labels": Segformer_B2_Fashion_Labels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SegformerB2ClothesUltraBatch": "Segformer B2 Clothes Ultra Fast",
    "Segformer_B2_Clothes_Labels": "Segformer B2 Clothes Label",
    "Segformer_B2_Clothes_Fashion_Labels": "Segformer B2 Fashion Label",
}
