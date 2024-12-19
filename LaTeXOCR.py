import re
import time
import traceback
from pathlib import Path
from typing import Tuple, List, Optional, Union
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
from io import BytesIO

import cv2
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

import chardet
import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE

InputType = Union[str, np.ndarray, bytes, Path]

class EncoderDecoder:
    def __init__(
        self,
        encoder_path: Union[Path, str],
        decoder_path: Union[Path, str],
        bos_token: int,
        eos_token: int,
        max_seq_len: int,
    ):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.max_seq_len = max_seq_len

        self.encoder = OrtInferSession(encoder_path)
        self.decoder = Decoder(decoder_path)

    def __call__(self, x: np.ndarray, temperature: float = 0.25):
        ort_input_data = np.array([self.bos_token] * len(x))[:, None]
        context = self.encoder([x])[0]
        output = self.decoder(
            ort_input_data,
            self.max_seq_len,
            eos_token=self.eos_token,
            context=context,
            temperature=temperature,
        )
        return output


class Decoder:
    def __init__(self, decoder_path: Union[Path, str]):
        self.max_seq_len = 512
        self.session = OrtInferSession(decoder_path)

    def __call__(
        self,
        start_tokens,
        seq_len=256,
        eos_token=None,
        temperature=1.0,
        filter_thres=0.9,
        context=None,
    ):
        num_dims = len(start_tokens.shape)

        b, t = start_tokens.shape

        out = start_tokens
        mask = np.full_like(start_tokens, True, dtype=bool)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len :]
            mask = mask[:, -self.max_seq_len :]

            ort_outs = self.session([x.astype(np.int64), mask, context])[0]
            np_preds = ort_outs
            np_logits = np_preds[:, -1, :]

            np_filtered_logits = self.npp_top_k(np_logits, thres=filter_thres)
            np_probs = self.softmax(np_filtered_logits / temperature, axis=-1)

            sample = self.multinomial(np_probs.squeeze(), 1)[None, ...]

            out = np.concatenate([out, sample], axis=-1)
            mask = np.pad(mask, [(0, 0), (0, 1)], "constant", constant_values=True)

            if (
                eos_token is not None
                and (np.cumsum(out == eos_token, axis=1)[:, -1] >= 1).all()
            ):
                break

        out = out[:, t:]
        if num_dims == 1:
            out = out.squeeze(0)
        return out

    @staticmethod
    def softmax(x, axis=None) -> float:
        def logsumexp(a, axis=None, b=None, keepdims=False):
            a_max = np.amax(a, axis=axis, keepdims=True)

            if a_max.ndim > 0:
                a_max[~np.isfinite(a_max)] = 0
            elif not np.isfinite(a_max):
                a_max = 0

            tmp = np.exp(a - a_max)

            # suppress warnings about log of zero
            with np.errstate(divide="ignore"):
                s = np.sum(tmp, axis=axis, keepdims=keepdims)
                out = np.log(s)

            if not keepdims:
                a_max = np.squeeze(a_max, axis=axis)
            out += a_max
            return out

        return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

    def npp_top_k(self, logits, thres=0.9):
        k = int((1 - thres) * logits.shape[-1])
        val, ind = self.np_top_k(logits, k)
        probs = np.full_like(logits, float("-inf"))
        np.put_along_axis(probs, ind, val, axis=1)
        return probs

    @staticmethod
    def np_top_k(
        a: np.ndarray, k: int, axis=-1, largest=True, sorted=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        if axis is None:
            axis_size = a.size
        else:
            axis_size = a.shape[axis]

        assert 1 <= k <= axis_size

        a = np.asanyarray(a)
        if largest:
            index_array = np.argpartition(a, axis_size - k, axis=axis)
            topk_indices = np.take(index_array, -np.arange(k) - 1, axis=axis)
        else:
            index_array = np.argpartition(a, k - 1, axis=axis)
            topk_indices = np.take(index_array, np.arange(k), axis=axis)

        topk_values = np.take_along_axis(a, topk_indices, axis=axis)
        if sorted:
            sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
            if largest:
                sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
            sorted_topk_values = np.take_along_axis(
                topk_values, sorted_indices_in_topk, axis=axis
            )
            sorted_topk_indices = np.take_along_axis(
                topk_indices, sorted_indices_in_topk, axis=axis
            )
            return sorted_topk_values, sorted_topk_indices
        return topk_values, topk_indices

    @staticmethod
    def multinomial(weights, num_samples, replacement=True):
        weights = np.asarray(weights)
        weights /= np.sum(weights)  # 确保权重之和为1
        indices = np.arange(len(weights))
        samples = np.random.choice(
            indices, size=num_samples, replace=replacement, p=weights
        )
        return samples

class LoadImage:
    def __init__(
        self,
    ):
        pass

    def __call__(self, img: InputType) -> np.ndarray:
        if not isinstance(img, InputType.__args__):
            raise LoadImageError(
                f"The img type {type(img)} does not in {InputType.__args__}"
            )

        img = self.load_img(img)
        img = self.convert_img(img)
        return img

    def load_img(self, img: InputType) -> np.ndarray:
        if isinstance(img, (str, Path)):
            self.verify_exist(img)
            try:
                img = np.array(Image.open(img))
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            img = np.array(Image.open(BytesIO(img)))
            return img

        if isinstance(img, np.ndarray):
            return img

        raise LoadImageError(f"{type(img)} is not supported!")

    def convert_img(self, img: np.ndarray):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3:
            channel = img.shape[2]
            if channel == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if channel == 2:
                return self.cvt_two_to_three(img)

            if channel == 4:
                return self.cvt_four_to_three(img)

            if channel == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            raise LoadImageError(
                f"The channel({channel}) of the img is not in [1, 2, 3, 4]"
            )

        raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")

    @staticmethod
    def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
        """RGBA → BGR"""
        r, g, b, a = cv2.split(img)
        new_img = cv2.merge((b, g, r))

        not_a = cv2.bitwise_not(a)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(new_img, new_img, mask=a)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def cvt_two_to_three(img: np.ndarray) -> np.ndarray:
        """gray + alpha → BGR"""
        img_gray = img[..., 0]
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        img_alpha = img[..., 1]
        not_a = cv2.bitwise_not(img_alpha)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(img_bgr, img_bgr, mask=img_alpha)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def verify_exist(file_path: Union[str, Path]):
        if not Path(file_path).exists():
            raise LoadImageError(f"{file_path} does not exist.")

class LoadImageError(Exception):
    pass

class OrtInferSession:
    def __init__(self, model_path: Union[str, Path], num_threads: int = -1):
        self.verify_exist(model_path)

        self.num_threads = num_threads
        self._init_sess_opt()

        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }
        EP_list = [(cpu_ep, cpu_provider_options)]
        try:
            self.session = InferenceSession(
                str(model_path), sess_options=self.sess_opt, providers=EP_list
            )
        except TypeError:
            # compatible with onnxruntime 1.5.2
            self.session = InferenceSession(str(model_path), sess_options=self.sess_opt)

    def _init_sess_opt(self):
        self.sess_opt = SessionOptions()
        self.sess_opt.log_severity_level = 4
        self.sess_opt.enable_cpu_mem_arena = False

        if self.num_threads != -1:
            self.sess_opt.intra_op_num_threads = self.num_threads

        self.sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    def __call__(self, input_content: List[np.ndarray]) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(None, input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_name(self, output_idx=0):
        return self.session.get_outputs()[output_idx].name

    def get_metadata(self):
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict

    @staticmethod
    def verify_exist(model_path: Union[Path, str]):
        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exist!")

        if not model_path.is_file():
            raise FileExistsError(f"{model_path} must be a file")


class ONNXRuntimeError(Exception):
    pass

class PreProcess:
    def __init__(self, max_dims: List[int], min_dims: List[int]):
        self.max_dims, self.min_dims = max_dims, min_dims
        self.mean = np.array([0.7931, 0.7931, 0.7931]).astype(np.float32)
        self.std = np.array([0.1738, 0.1738, 0.1738]).astype(np.float32)

    @staticmethod
    def pad(img: Image.Image, divable: int = 32) -> Image.Image:
        """Pad an Image to the next full divisible value of `divable`. Also normalizes the image and invert if needed.

        Args:
            img (PIL.Image): input image
            divable (int, optional): . Defaults to 32.

        Returns:
            PIL.Image
        """
        threshold = 128
        data = np.array(img.convert("LA"))
        if data[..., -1].var() == 0:
            data = (data[..., 0]).astype(np.uint8)
        else:
            data = (255 - data[..., -1]).astype(np.uint8)

        data = (data - data.min()) / (data.max() - data.min()) * 255
        if data.mean() > threshold:
            # To invert the text to white
            gray = 255 * (data < threshold).astype(np.uint8)
        else:
            gray = 255 * (data > threshold).astype(np.uint8)
            data = 255 - data

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        rect = data[b : b + h, a : a + w]
        im = Image.fromarray(rect).convert("L")
        dims: List[Union[int, int]] = []
        for x in [w, h]:
            div, mod = divmod(x, divable)
            dims.append(divable * (div + (1 if mod > 0 else 0)))

        padded = Image.new("L", tuple(dims), 255)
        padded.paste(im, (0, 0, im.size[0], im.size[1]))
        return padded

    def minmax_size(
        self,
        img: Image.Image,
    ) -> Image.Image:
        """Resize or pad an image to fit into given dimensions

        Args:
            img (Image): Image to scale up/down.

        Returns:
            Image: Image with correct dimensionality
        """
        if self.max_dims is not None:
            ratios = [a / b for a, b in zip(img.size, self.max_dims)]
            if any([r > 1 for r in ratios]):
                size = np.array(img.size) // max(ratios)
                size = np.maximum(size, 1)
                img = img.resize(tuple(size.astype(int)), Image.BILINEAR)

        if self.min_dims is not None:
            padded_size: List[Union[int, int]] = [
                max(img_dim, min_dim)
                for img_dim, min_dim in zip(img.size, self.min_dims)
            ]

            new_pad_size = tuple(padded_size)
            if new_pad_size != img.size:  # assert hypothesis
                padded_im = Image.new("L", new_pad_size, 255)
                padded_im.paste(img, img.getbbox())
                img = padded_im
        return img

    def normalize(self, img: np.ndarray, max_pixel_value=255.0) -> np.ndarray:
        mean = self.mean * max_pixel_value
        std = self.std * max_pixel_value
        denominator = np.reciprocal(std, dtype=np.float32)
        img = img.astype(np.float32)
        img -= mean
        img *= denominator
        return img

    @staticmethod
    def to_gray(img) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def transpose_and_four_dim(img: np.ndarray) -> np.ndarray:
        return img.transpose(2, 0, 1)[:1][None, ...]

        
class TokenizerCls:
    def __init__(self, json_file: Union[Path, str]):
        self.tokenizer = Tokenizer(BPE()).from_file(str(json_file))
    
    def token2str(self, tokens) -> List[str]:
        if len(tokens.shape) == 1:
            tokens = tokens[None, :]
    
        dec = [self.tokenizer.decode(tok.tolist()) for tok in tokens]
        return [
            "".join(detok.split(" "))
            .replace("Ġ", " ")
            .replace("[EOS]", "")
            .replace("[BOS]", "")
            .replace("[PAD]", "")
            .strip()
            for detok in dec
        ]
        
class LaTeXOCR:
    def __init__(
        self,
        image_resizer_path: Union[str, Path] = None,
        encoder_path: Union[str, Path] = None,
        decoder_path: Union[str, Path] = None,
        tokenizer_json: Union[str, Path] = None,
    ):
        self.image_resizer_path = image_resizer_path
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.tokenizer_json = tokenizer_json
        
        input_params = {
    'max_width': 672, 'max_height': 192, 'min_height': 32, 'min_width': 32, 'bos_token': 1, 'max_seq_len': 512, 'eos_token': 2, 'temperature': 0.00001
    }
        self.max_dims = [input_params['max_width'], input_params['max_height']]
        self.min_dims = [input_params['min_width'], input_params['min_height']]
        self.temperature = input_params['temperature']

        self.load_img = LoadImage()
        self.pre_pro = PreProcess(max_dims=self.max_dims, min_dims=self.min_dims)
        self.image_resizer = OrtInferSession(self.image_resizer_path)

        self.encoder_decoder = EncoderDecoder(
            encoder_path=self.encoder_path,
            decoder_path=self.decoder_path,
            bos_token=input_params['bos_token'],
            eos_token=input_params['eos_token'],
            max_seq_len=input_params['max_seq_len'],
        )
        self.tokenizer = TokenizerCls(self.tokenizer_json)

    def __call__(self, img: InputType) -> Tuple[str, float]:
        s = time.perf_counter()

        try:
            img = self.load_img(img)
        except LoadImageError as exc:
            error_info = traceback.format_exc()
            raise LoadImageError(
                f"Load the img meets error. Error info is {error_info}"
            ) from exc

        try:
            resizered_img = self.loop_image_resizer(img)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ValueError(
                f"image resizer meets error. Error info is {error_info}"
            ) from e

        try:
            dec = self.encoder_decoder(resizered_img, temperature=self.temperature)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ValueError(
                f"EncoderDecoder meets error. Error info is {error_info}"
            ) from e

        decode = self.tokenizer.token2str(dec)
        pred = self.post_process(decode[0])

        elapse = time.perf_counter() - s
        return pred, elapse

    def loop_image_resizer(self, img: np.ndarray) -> np.ndarray:
        pillow_img = Image.fromarray(img)
        pad_img = self.pre_pro.pad(pillow_img)
        input_image = self.pre_pro.minmax_size(pad_img).convert("RGB")
        r, w, h = 1, input_image.size[0], input_image.size[1]
        for _ in range(10):
            h = int(h * r)
            final_img, pad_img = self.pre_process(input_image, r, w, h)

            resizer_res = self.image_resizer([final_img.astype(np.float32)])[0]

            argmax_idx = int(np.argmax(resizer_res, axis=-1))
            w = (argmax_idx + 1) * 32
            if w == pad_img.size[0]:
                break

            r = w / pad_img.size[0]
        return final_img

    def pre_process(
        self, input_image: Image.Image, r, w, h
    ) -> Tuple[np.ndarray, Image.Image]:
        if r > 1:
            resize_func = Image.Resampling.BILINEAR
        else:
            resize_func = Image.Resampling.LANCZOS

        resize_img = input_image.resize((w, h), resize_func)
        pad_img = self.pre_pro.pad(self.pre_pro.minmax_size(resize_img))
        cvt_img = np.array(pad_img.convert("RGB"))

        gray_img = self.pre_pro.to_gray(cvt_img)
        normal_img = self.pre_pro.normalize(gray_img)
        final_img = self.pre_pro.transpose_and_four_dim(normal_img)
        return final_img, pad_img

    @staticmethod
    def post_process(s: str) -> str:
        """Remove unnecessary whitespace from LaTeX code.

        Args:
            s (str): Input string

        Returns:
            str: Processed image
        """
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = r"[\W_^\d]"
        names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
        return s