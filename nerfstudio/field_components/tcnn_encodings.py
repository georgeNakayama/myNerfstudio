import torch 
import warnings
from tinycudann.modules import _torch_precision, _C, _module_function

class NoGradModule(torch.nn.Module):
    def __init__(self, seed=1337):
        super(NoGradModule, self).__init__()

        self.native_tcnn_module = self._native_tcnn_module()
        self.dtype = _torch_precision(self.native_tcnn_module.param_precision())

        self.seed = seed
        initial_params = self.native_tcnn_module.initial_params(seed)
        self.params = initial_params

        self.loss_scale = 128.0 if self.native_tcnn_module.param_precision() == _C.Precision.Fp16 else 1.0


    def forward(self, x):
        if not x.is_cuda:
            warnings.warn("input must be a CUDA tensor, but isn't. This indicates suboptimal performance.")
            x = x.cuda()

        batch_size = x.shape[0]
        batch_size_granularity = int(_C.batch_size_granularity())
        padded_batch_size = (batch_size + batch_size_granularity-1) // batch_size_granularity * batch_size_granularity

        x_padded = x if batch_size == padded_batch_size else torch.nn.functional.pad(x, [0, 0, 0, padded_batch_size - batch_size])
        output = _module_function.apply(
            self.native_tcnn_module,
            x_padded.to(torch.float).contiguous(),
            self.params.to(_torch_precision(self.native_tcnn_module.param_precision())).contiguous(),
            self.loss_scale
        )
        return output[:batch_size, :self.n_output_dims]

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        # Avoid pickling native objects
        del state["native_tcnn_module"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reconstruct native entries
        self.native_tcnn_module = self._native_tcnn_module()

    def extra_repr(self):
        return f"n_input_dims={self.n_input_dims}, n_output_dims={self.n_output_dims}, seed={self.seed}, dtype={self.dtype}, hyperparams={self.native_tcnn_module.hyperparams()}"

class NoGradEncoding(NoGradModule):
	"""
	Input encoding to a neural network.

	Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
	it to a `dtype` tensor of shape `[:, self.n_output_dims]`, where
	`self.n_output_dims` depends on `n_input_dims` and the configuration
	`encoding_config`.

	Parameters
	----------
	n_input_dims : `int`
		Determines the shape of input tensors as `[:, n_input_dims]`
	encoding_config: `dict`
		Configures the encoding. Possible configurations are documented at
		https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
	seed: `int`
		Seed for pseudorandom parameter initialization
	dtype: `torch.dtype`
		Precision of the output tensor and internal parameters. A value
		of `None` corresponds to the optimally performing precision,
		which is `torch.half` on most systems. A value of `torch.float`
		may yield higher numerical accuracy, but is generally slower.
		A value of `torch.half` may not be supported on all systems.
	"""
	def __init__(self, n_input_dims, encoding_config, seed=1337, dtype=None):
		self.n_input_dims = n_input_dims
		self.encoding_config = encoding_config
		if dtype is None:
			self.precision = _C.preferred_precision()
		else:
			if dtype == torch.float32:
				self.precision = _C.Precision.Fp32
			elif dtype == torch.float16:
				self.precision = _C.Precision.Fp16
			else:
				raise ValueError(f"Encoding only supports fp32 or fp16 precision, but got {dtype}")

		super(NoGradEncoding, self).__init__(seed=seed)

		self.n_output_dims = self.native_tcnn_module.n_output_dims()

	def _native_tcnn_module(self):
		return _C.create_encoding(self.n_input_dims, self.encoding_config, self.precision)
