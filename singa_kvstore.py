from singa import autograd
from singa import tensor
from singa import opt
from singa.proto import core_pb2
import mxnet as mx
import SingaOpt
#---------start ps API---------------------------------
def create_kvstore(kv_type, opt_type,**kargs):
    kv_ret = mx.kv.create(kv_type) #create kv store for ps
    kv_ret.set_optimizer(mx.optimizer.create(opt_type,**kargs))
    return kv_ret


def tensor2numpy_nocopy(th):
    '''Copy the tensor into a numpy array by sharing the same memory.

    Args:
        t (Tensor): a Tensor

    Returns:
        a numpy array
    '''
    if th.dtype == core_pb2.kFloat32:
        np_array = th.data.GetFloatValue(int(th.size()))
    elif th.dtype == core_pb2.kInt:
        np_array = th.data.GetIntValue(int(th.size()))
    else:
        print('Not implemented yet for ', th.dtype)
    return np_array.reshape(th.shape)

is_kvInitial = False
def backward_and_update(kv,loss):
    global is_kvInitial
    if is_kvInitial != True:
       #Initial kv store for workers of ps-architecture
       key = 0
       for p, g in autograd.backward(loss):
           kv.init(key, mx.nd.from_numpy(tensor2numpy_nocopy(p),zero_copy=True))
           key += 1  
       is_kvInitial = True
    else:     
       #push
       kv_pairs = []
       key = 0
       for p, g in autograd.backward(loss):
           #print('tensor.data.type is %s' % type(p.data).__name__)
           kv.push(key,mx.nd.from_numpy(tensor2numpy_nocopy(g),zero_copy=True))
           kv_pairs.append((key,p,g))
           key += 1
       #pull
       for key,p,g in kv_pairs:
           #out_buf = mx.nd.zeros(p.shape)
           #print('before pull->p')
           #print(tensor2numpy_nocopy(p)[0])
           out_buf = mx.nd.from_numpy(tensor2numpy_nocopy(p),zero_copy=True)
           kv.pull(key,out=out_buf)
           #print("after pull->p")
           #print(tensor2numpy_nocopy(p)[0])
           p.copy_from_numpy(out_buf.asnumpy())
           #print("after copy->p")
           #print(tensor2numpy_nocopy(p)[0])
           
#--------end ps API---------------------------------

