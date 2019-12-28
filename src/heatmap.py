import innvestigate
import innvestigate.utils

import numpy as np
import cv2
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class LRP():
    def __init__(self,model,preprocess,modelType='InceptionV3',strategy="lrp.alpha_1_beta_0_IB"):
        self.strategy = strategy
        self.model = model
        self.modelType = modelType
        # if (modelType == 'VGG16'):
        #     self.strategy = 'lrp.alpha_1_beta_0_IB'
        if(modelType=='InceptionV3' or modelType =='ResNet50' or modelType == 'DaveII'):
            self.strategy = 'deep_taylor'
        self.model_noSoftMax = innvestigate.utils.model_wo_softmax(model) # strip the softmax layer
        self.analyzer = innvestigate.create_analyzer(self.strategy, self.model_noSoftMax)
        self.preprocess = preprocess
    def generate_heatmap(self,x):
        if (type(x) == str):
            x = cv2.imread(x)
        x = self.preprocess(x)
        a = self.analyzer.analyze(x)
        a = a[0]
        print(a)
        a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
        a /= np.max(np.abs(a))+1e-6
        a = (a*255).astype(np.uint8)
        print(a)
        #heatmapshow = cv2.normalize(a, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(a, cv2.COLORMAP_JET)
        heatmapshow = heatmapshow.astype(np.float64)
        cv2.imshow('heat',heatmapshow.astype(np.uint8))
        cv2.waitKey(1000)
        #heatmapshow = heatmapshow.sum(axis=np.argmax(np.asarray(heatmapshow.shape) == 3))
        #heatmapshow /= np.max(np.abs(heatmapshow))
        return heatmapshow

# if __name__ == "__main__":
#     import os
#     from generate_model import generate_model
#     files = os.listdir('../ignoreme//dogs')
#     paths = [os.path.join('../ignoreme/dogs',file) for file in files]
#     model_generator = generate_model('InceptionV3')
#     lrp = LRP(model,modelType=model_generator.modelType)
#     for path in paths:
#         x = lrp.generate_heatmap(path)
#         name = os.path.basename(path)
#         cv2.imwrite('../ignoreme/dogs/heatmap_inception_'+name, x)

#KeyError: "No analyzer with the name 'lrp.lrp.sequential_preset_a' could be found. All possible n
# ames are: ['input', 'random', 'gradient', 'gradient.baseline', 'input_t_gradient', 'deconvnet', 'guided_backprop',
# 'integrated_gradients', 'smoothgrad', 'lrp', 'lrp.z', 'lrp.z_IB', 'lrp.epsilon', 'lrp.epsilon_IB', 'lrp.w_square',
# 'lrp.flat', 'lrp.alpha_beta', 'lrp.alpha_2_beta_1', 'lrp.alpha_2_beta_1_IB', 'lrp.alpha_1_beta_0', 'lrp.alpha_1_beta_0_IB',
# 'lrp.z_plus', 'lrp.z_plus_fast', 'lrp.sequential_preset_a', 'lrp.sequential_preset_b', 'lrp.sequential_preset_a_flat', 'lrp.sequential_preset_b_flat', 'deep_taylor', 'deep_taylor.bounded', 'deep_lift.wrapper', 'pattern.net', 'pattern.attribution']"
