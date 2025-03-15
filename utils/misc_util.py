import clip
import torch

def have_overlap(seg1, seg2):
    if seg1[0] > seg2[1] or seg2[0] > seg1[1]:
        return False
    else:
        return True


def get_overlap(seg1, seg2):
    overlap_len = max(0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]))
    return overlap_len


def load_and_freeze_clip(clip_version, device='cpu'):
    clip_model, clip_preprocess = clip.load(clip_version, device=device,
                                            jit=False)  # Must set jit=False for training
    clip.model.convert_weights(
        clip_model)  # Actually this line is unnecessary since clip by default already on float16

    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model


def encode_text(clip_model, raw_text, force_empty_zero=True):
    device = next(clip_model.parameters()).device
    # raw_text - list (batch_size length) of strings with input text prompts
    texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length]
    text_embedding = clip_model.encode_text(texts).float() # [bs, 512]
    if force_empty_zero:  # force empty string to have zero embedding, same as being masked out in original MDM
        empty_text = [text == '' for text in raw_text]
        text_embedding[empty_text, :] = 0
    return text_embedding


def compose_texts_with_and(texts):
    texts = sorted(texts)
    return ' and '.join(texts)

def dict_to_args(dict_args):
    from dataclasses import make_dataclass, asdict
    dynamic_class = make_dataclass('DynamicMotionModelArgs', fields=[(key, type(dict_args[key])) for key in dict_args])
    args = dynamic_class(**dict_args)

    return args
