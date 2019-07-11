import p1_nn as p1
import numpy as np

def test_CE():
    print("test CE")
    eps = 0.2
    y = 10*np.random.rand(10,1)
    labels = np.random.rand(10,1)
    for i in range(0,len(labels)):
        if labels[i]>0.5:
            labels[i] = 1
        else:
            labels[i] = 0
 #   print("y: "+str(y)+" labels: "+str(labels))
    cost = p1.forward_cross_entropy_loss(y, labels)
    grad_CE_Loss = p1.backward_cross_entropy_loss(y, labels)
    y_a = np.zeros_like(y)
    y_a[0] = eps
    y_aftera_ = y + y_a
    cost_inputplusa_ = p1.forward_cross_entropy_loss(y_aftera_, labels)
#    print("costs: "+str(cost)+" "+str(cost_inputplusa_))
    CE_gradient_check = (cost_inputplusa_ - cost) / eps
#    print('Check first element of grad_CE_Loss and linear approximation of gradient CE Loss')
    print(CE_gradient_check)
    print(grad_CE_Loss)


    
def test_softmax():
    print("test softmax")
    eps = 0.00002
    x = 10*np.random.rand(10,1)
    #print("x: "+str(x)+" shape: "+str(x.shape))
    labels = np.random.rand(10,1)
    for i in range(0,len(labels)):
        if labels[i]>0.5:
            labels[i] = 1
        else:
            labels[i] = 0
    x_a = np.zeros_like(x)
    x_a[0] = eps
#    x_a[2] = eps
    x_aftera_ = x + x_a
    
    y_plusa_ = p1.forward_softmax(x_aftera_)
    ce_plusa = p1.forward_cross_entropy_loss(y_plusa_,labels)
    gradloss_plusa = p1.backward_cross_entropy_loss(y_plusa_,labels)
    grad_softmax_plusa = p1.backward_softmax(x,gradloss_plusa)
 
    y = p1.forward_softmax(x)
    ce = p1.forward_cross_entropy_loss(y,labels)
    grad_loss = p1.backward_cross_entropy_loss(y,labels)
    grad_loss_softmax = p1.backward_softmax(x,grad_loss)
    
    gradcheck = (ce_plusa - ce)/eps
    print("costs: "+str(ce_plusa)+" "+str(ce))   
#    gradcheck_sm = (y_plusa_-y)/eps
    print("actual: "+str(gradcheck))
    print("grad: :"+str(grad_loss_softmax))
#   print("costs: "+str(ce)+" "+str(ce_plusa))
    #print("grad_loss: "+str(grad_loss))
#    print("grad loss softmax: "+str(grad_loss_softmax))
    #print("grad check: "+str(gradcheck_sm))

#    print("check soft: "+str(gradcheck_sm))


def test_relu():
    print("test relu")

def test_linear():
    eps = 0.002
    x = 10*np.random.rand(10,)
    W = 10* np.random.rand(10,6)
    b = 10*np.random.rand(6)
    flin = p1.forward_linear(W, b,x)
    print("flin: "+str(flin))
    x_a = np.zeros_like(x)
    x_a[0] = eps
    x_aftera_ = x+x_a
    flin_aftera = p1.forward_linear(W,b,x_aftera_)
    testgrad = np.zeros(6)
    testgrad[0] = 1
    backlinW,backlinb,backlinx = p1.backward_linear(W,b,x,testgrad)
    print("backlinx: "+str(backlinx))
    testgrad = (flin_aftera - flin)/eps
    print("test grad: "+str(testgrad))

test_CE()
test_softmax()
#test_linear()
#test_relu()


