import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim, ssim, ssim as f_ssim
from torch.autograd import Variable
from torch.fft import fft2, ifft2
from torchvision.models import vgg19, VGG19_Weights

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(Variable(torch.sqrt(diff * diff + self.epsilon * self.epsilon).type(torch.FloatTensor), requires_grad=True))
        return loss

class MSEGDL(nn.Module):
    def __init__(self, lambda_mse=1, lambda_gdl=1):
        super(MSEGDL, self).__init__()
        self.lambda_mse = lambda_mse
        self.lambda_gdl = lambda_gdl

    def forward(self, inputs, targets):

        squared_error = (inputs - targets).pow(2)
        gradient_diff_i = (inputs.diff(axis=-1)-targets.diff(axis=-1)).pow(2)
        gradient_diff_j =  (inputs.diff(axis=-2)-targets.diff(axis=-2)).pow(2)
        loss = (self.lambda_mse*squared_error.sum() + self.lambda_gdl*gradient_diff_i.sum() + self.lambda_gdl*gradient_diff_j.sum())/inputs.numel()

        return loss

class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, size_average=True):
        super(SSIMLoss, self).__init__()
        self.data_range = data_range
        self.size_average = size_average

    def forward(self, img1, img2):
        return 1 - Variable(ssim(img1, img2, data_range=self.data_range, size_average=self.size_average).type(torch.FloatTensor), requires_grad=True)

class MSSSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, size_average=True):
        super(MSSSIMLoss, self).__init__()
        self.data_range = data_range
        self.size_average = size_average

    def forward(self, img1, img2):
        return 1 - Variable(ms_ssim(img1, img2, data_range=self.data_range, size_average=self.size_average).type(torch.FloatTensor), requires_grad=True)

class VGGLoss(nn.Module):
    def __init__(self, layer=36):
        super().__init__()

        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:layer].eval()
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        self.vgg.eval()
        vgg_input_features = self.vgg(output)
        vgg_target_features = self.vgg(target)
        loss = self.loss(vgg_input_features, vgg_target_features)
        del vgg_input_features, vgg_target_features
        gc.collect()
        torch.cuda.empty_cache()
        return loss

class DeblurLoss(nn.Module):
    """Advanced loss function combining multiple objectives"""
    def __init__(self):
        super(DeblurLoss, self).__init__()
        self.l1_loss   = nn.L1Loss()
        self.mse_loss  = nn.MSELoss()
        self.gdl_loss  = MSEGDL()
        self.ssim_loss = SSIMLoss()
        self.vgg_loss  = VGGLoss()
            
    def get_frequency_loss(self, pred, target):
        # FFT-based frequency loss
        pred_freq = torch.fft.fft2(pred)
        target_freq = torch.fft.fft2(target)
        return F.mse_loss(pred_freq.abs(), target_freq.abs())

    def forward(self, pred_list, target):
        total_loss = 0
        weights = [1.0, 0.75, 0.45, 0.35]  # Weights for different scales
        
        for pred, weight in zip(pred_list, weights):
            # Resize target to match prediction if needed
            if pred.shape != target.shape:
                target_resized = F.interpolate(target, size=pred.shape[2:]).to(target)
            else:
                target_resized = target

            pred = pred.to(target)
            # Pixel loss
            pixel_loss = self.l1_loss(pred, target_resized)
            
            # Frequency loss
            freq_loss = self.get_frequency_loss(pred, target_resized)
            
            # Perceptual loss
            perc_loss = self.vgg_loss(pred, target_resized)

            # SSIM loss
            ssim_loss = self.ssim_loss(pred, target_resized)

            # GDL loss
            gdl_loss = self.gdl_loss(pred, target_resized)
            
            # Combine losses with weights
            total_loss += weight * (
                1.0 * pixel_loss + 
                0.6 * ssim_loss +
                0.3 * gdl_loss +
                0.1 * freq_loss + 
                0.8 * perc_loss
            )
            
        return total_loss