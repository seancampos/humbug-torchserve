# +
import io
# model_handler.py

"""
ModelHandler defines a custom model handler.
"""
import logging

import itertools
#from ts.torch_handler.base_handler import BaseHandler
from ts.torch_handler.image_classifier import ImageClassifier
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as AT

logger = logging.getLogger(__name__)

class ModelHandler(ImageClassifier):
    """
    A custom model handler implementation.
    """
    def initialize(self, context):
        super().initialize(context)
        
        self.set_max_result_classes(2)
    
    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        
        The preprocess function of MNIST program converts the input data to a float tensor
        Args:
            data (List): Input data from the request is in the form of a Tensor
        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        
        sr = 8000
        waveforms = []
        

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            recording = row.get("data") or row.get("body")
            if isinstance(recording, str):
                # if the image is a string of bytesarray.
                recording = base64.b64decode(recording)

            # If the image is sent as bytesarray
#             if isinstance(recording, (bytearray, bytes)):
#                 logger.info("wave")
#                 waveform, inp_rate = torchaudio.load(io.BytesIO(recording))
#                 if inp_rate != sr:
#                     resampler = AT.Resample(inp_rate, sr, dtype=waveform.dtype)
#                     waveform = resampler(waveform)
#                 waveforms.append(waveform[0,:int(1.92 * sr)].unsqueeze(0))
#                 # offsets = range(0,2)
#                 # for offset in offsets:
#                 #     waveforms.append(waveform[0, int(offset*(1.92 * sr)): int((offset + 1) * (1.92 * sr))])
#             else:
            logger.info("tensor")
                # if the image is a list
            waveform = torch.FloatTensor(recording)

            waveforms.append(waveform)

        return torch.stack(waveforms).to(self.device)
        
        # Take the input data and make it inference ready
#         preprocessed_data = data[0].get("data")
        
#         if preprocessed_data is None:
#             preprocessed_data = data[0].get("body")
            
#         waveform, inp_rate = torchaudio.load(preprocessed_data)
#         if inp_rate != 8000:
#             resampler = AT.Resample(inp_rate, sr, dtype=waveform.dtype)
#             waveform = resampler(waveform)
        

#         return waveform[0,:int(1.92*8000)].unsqueeze(0)
    
    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.
        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.
        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        marshalled_data = data.to(self.device)
        with torch.no_grad():
            results = self.model(marshalled_data, *args, **kwargs)['prediction']
        return results

    # def postprocess(self, data):
    #     ps = F.softmax(data["predicition"], dim=1)
    #     probs, classes = torch.topk(ps, self.topk, dim=1)
    #     probs = probs.tolist()
    #     classes = classes.tolist()

    #     lbl_classes = itertools.repeat(range(len(probs[0])), len(probs))

    #     results = [
    #         {
    #             (str(lbl_class)): prob
    #             for lbl_class, prob in zip(*row)
    #         }
    #         for row in zip(lbl_classes, probs)
    #     ]
    #     return results.append({"spectrogram": data["spectrogram"]})