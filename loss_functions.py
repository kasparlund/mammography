import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#def maskmult(class_mask, class_images):
#    "activate or deactivate a class according to the mask for the class (0 1)"
#    assert(len(class_mask)==len(class_images))
#    for i in range(len(class_mask)):
#          class_images[i] *= class_mask[i]
            

def torch_weigths_add_noise( scale, w ):
    #add uniform noise at scale to non zero w
    wn = w + scale*torch.rand(len(w)).cuda()
    wn[w.le(0)] = 0
    return wn/wn.sum()

def torch_weigths_mult_noise( scale, w ):
    #add uniform noise at scale to non zero w
    s  = 1.0 + scale*2.*( torch.rand(len(w)).cuda()-0.5)  
    w  = w*s
    return w/w.sum()


weights = torch.from_numpy( code_weights ).cuda()
ix_weights = torch.from_numpy( (code_weights>0).astype(np.long) ) 

class DiceLoss(nn.Module):
    dicemsg=""
    def __init__( self):
        super(DiceLoss, self).__init__()
    
    @staticmethod
    def dice_loss_1ch(input, target):
        smooth = 1.
        input = torch.sigmoid(input)
        iflat = input.view(-1)
        tflat = target.view(-1)
        #DiceLoss.dicemsg = (f"shape of input:{input.shape}, target:{target.shape} \
        #                     input.type():{input.type()} iflat.type():{iflat.type()}  tflat.type():{tflat.type()}")
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
        
    @staticmethod
    def dice_loss_nch(input, target):
        dlc = 0
        bs, n_classes = input.shape[:2] 
        target = target.squeeze().float()
        n_real_classes=0
        for i in range(n_classes):
            dlc += DiceLoss.dice_loss_1ch(input[:,i], target)
        return dlc/n_classes/bs
    def forward(self, input, target):
        input  = input[:,ix_weights]
        #DiceLoss.dicemsg = (f"shape of input:{input.shape} of target:{target.shape}")
        return DiceLoss.dice_loss_nch(input,target)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(np.float32)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        input  = input[:,ix_weights]
        
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()    
        
class FocalLoss_lafoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ( (-max_val).exp() + (-input - max_val).exp() ).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()    
    
def acc_lafoss(preds,targs,th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()    


def make_one_hot(labels, C=2):
    one_hot = w(torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_())
    target = one_hot.scatter_(1, labels.data, 1)
    
    target = w(Variable(target))
        
    return target

class FocalLossMultiLabel(nn.Module):
    def __init__(self, gamma, weight):
        super().__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(weight=weight, reduce=False)
        
    def forward(self, input, target):
        loss = self.nll(input, target)
        
        one_hot = make_one_hot(target.unsqueeze(dim=1), input.size()[1])
        inv_probs = 1 - input.exp()
        focal_weights = (inv_probs * one_hot).sum(dim=1) ** self.gamma
        loss = loss * focal_weights
        
        return loss.mean()