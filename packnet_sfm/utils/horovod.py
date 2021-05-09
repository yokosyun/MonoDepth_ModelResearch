try:
    import horovod.torch as hvd

    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


# def reduce_value(value, average, name):
#     """
#     Reduce the mean value of a tensor from all GPUs

#     Parameters
#     ----------
#     value : torch.Tensor
#         Value to be reduced
#     average : bool
#         Whether values will be averaged or not
#     name : str
#         Value name

#     Returns
#     -------
#     value : torch.Tensor
#         reduced value
#     """
#     return hvd.allreduce(value, average=average, name=name)
