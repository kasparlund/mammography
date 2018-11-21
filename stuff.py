def noise_zm( scale, w ):
    s = 1+scale*(np.random.rand(len(w))-.5)*2
    w *= s
    return s/s.sum()
def noise_p( scale, size ):
    return scale*np.random.rand(size)
def noise_weigths( scale, w ):
    #add uniform noise at scale to non zero w
    wn = w + scale*np.random.rand(len(w))*(w>0)
    return wn/wn.sum()

def torch_weigths_add_noise( scale, w ):
    #add uniform noise at scale to non zero w
    wn = w + scale*torch.rand(len(w))
    wn[w.le(0)] = 0
    return wn/wn.sum()

def torch_weigths_mult_noise( scale, w ):
    #add uniform noise at scale to non zero w
    s  = 1.0 + scale*2.*( torch.rand(len(w))-0.5)  
    w  = w*s
    return w/w.sum()

w = np.zeros(5)+1
s = noise_zm(.2,w)

print(f"s:{s}")
print(f"mean,sd: {s.mean()}  {s.std()}  {s.min()}  {s.max()}")

scale = 0.5
w  = noise_weigths( scale, code_weights )
ix = w>0.

wt = torch.from_numpy(code_weights.astype(np.float32))
w2 = torch_weigths_mult_noise( scale, wt  )
ixt = w2>0.
print(w2)

#print(.2 * (np.random.ranf()-.5) + 1)
print(f"code_weights:{code_weights}" )
print(f"           w:{w2}" )
print(f"          w2:{w2/(1e-6+wt)}" )
print(f"code_weights   -mean,sd: {code_weights.mean()}  {code_weights.std()}  {code_weights.min()}  {code_weights.max()}")
print(f"code_weights+s -mean,sd: {w[ix].mean()}  {w[ix].std()}  {w[ix].min()}  {w[ix].max()}")
print(f"code_weights+s2-mean,sd: {w2[ixt].mean()}  {w2[ixt].std()}  {w2[ixt].min()}  {w2[ixt].max()}")


"""
def maskassign( class_images, dim, class_mask, value=0):
    "activate or deactivate a class according to the mask for the class (0 1)"
    assert(dim==0 or dim==1) 
    if dim==0:
        assert(len(class_mask)==class_images.size()[0]) 
        for i in range(len(class_mask)):
            if class_mask[i] <= 0. :
                class_images[i] = value
    elif dim==1:
        assert(len(class_mask)==class_images.size()[1]) 
        for i in range(len(class_mask)):
            if class_mask[i] <= 0. :
                class_images[:,i] = value


def maskmult( class_images, dim, class_mask):
    "activate or deactivate a class according to the mask for the class (0 1)"
    assert(dim==0 or dim==1) 
    if dim==0:
        assert(len(class_mask)==class_images.size()[0]) 
        for i in range(len(class_mask)):
              class_images[i] *= class_mask[i] 
    elif dim==1:
        assert(len(class_mask)==class_images.size()[1]) 
        for i in range(len(class_mask)):
              class_images[:,i] *= class_mask[i]                



#torch.cuda.empty_cache()
gc.collect()

bs   = [16,4,2] 
size = np.asarray([1, 2, 3])*224
lr   = [0.001, 0.005, 0.01]
msg = []
learn = None
for i in range(0,len(lr)):
    data = ImageDataBunch.create(dsTrain, dsValid, ds_tfms=tfms, tfm_y=True, bs=bs[i], size=size[i] )
    data.normalize(imagenet_stats)

    if learn is None : 
        learn = Learner.create_unet(data, models.resnet50, loss_func= WeighedCrossEntropy(code_weights), 
                                    metrics=accuracy_)
    else: 
        learn.data = data
    
    train_ds_im_size,valid_ds_im_size = learn.data.train_ds.kwargs["size"], learn.data.valid_ds.kwargs["size"]
    msg.append( f"batch size:{bs[i]} image size for train_ds { train_ds_im_size} - valid_ds for {valid_ds_im_size}" )
    
    torch.cuda.empty_cache()
    gc.collect()
    learn.fit_one_cycle(len(lr)-i, max_lr=lr[i])

msg

printWeight(learn.model)
weightsBeforeUnfreeze =  WeightsAsImage(learn.model)
weightsBeforeUnfreeze.shape

#learn.unfreeze()
#unfreeze first layer
layers = getLayers(learn.model)
for param in layers[0].parameters():
    param.requires_grad = True
    
weightsAfterTrainingInputLayer = WeightsAsImage(learn.model)
fig,axes = plt.subplots(nrows=1, ncols=2, figsize =(12,12), dpi=100)
axes[0].imshow(weightsBeforeUnfreeze)  
axes[1].imshow(weightsAfterTrainingInputLayer)  

diff = weightsAfterTrainingInputLayer-weightsBeforeUnfreeze
diff = (diff-diff.min())/(diff.max()-diff.min())
plotTensor(diff)
weightsAfterTrainingInputLayer-weightsBeforeUnfreeze
"""

