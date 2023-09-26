import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .models.basicnet import basic_net
from .models.osnet import *

class Extractor(object):
    __model_factory = {
    "basic_net": basic_net,
    "osnet_x1_0": osnet_x1_0,
    "osnet_x0_75": osnet_x0_75,
    "osnet_x0_5": osnet_x0_5,
    "osnet_x0_25": osnet_x0_25
    }
    def __init__(self, name_extractor, pretrained=True, use_cuda=True):
        self.model = self.__init_model(name_extractor, pretrained=pretrained)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.eval()
        # state_dict = torch.load(pretrain_path, map_location=lambda storage, loc: storage)['net_dict']
        # self.model.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        if pretrained:
            logger.info(f"{name_extractor} Loading weights from pretrained .... Done!" )
        else:
            logger.info(f"{name_extractor} INIT WITHOUT USING PRETRAIN .... Done!" )
        self.model.to(self.device)
        # self.size = (64, 128) # (w, h)
        self.size = (128, 256)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
    def __init_model(self, name, pretrained = True):
        avai_models = list(self.__model_factory.keys())
        if name not in avai_models:
            raise KeyError(
                'Unknown model: {}. Must be one of {}'.format(name, avai_models)
            )
        return self.__model_factory[name](pretrained=pretrained)

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            # im = cv2.resize(im.astype(np.float32)/255., (size))
            # cv2.imshow("abs",im)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.model(im_batch)
            print(features.cpu().numpy().shape)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

