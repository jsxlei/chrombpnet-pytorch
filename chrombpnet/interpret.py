# Author: Lei Xiong <jsxlei@gmail.com>

from tangermeme.deep_lift_shap import _nonlinear, _captum_deep_lift_shap
from tangermeme.utils import _validate_input

from .model_wrappers import ProfileWrapper, CountWrapper, _ProfileLogitScaling, _Log, _Exp
from .data_utils import get_seq, load_region_df, hdf5_to_bigwig, html_to_pdf

import pandas as pd
import numpy as np
import os
import torch
import pyfaidx

from .genome import motifs_datasets, hg38
MEME_FILE = motifs_datasets().fetch("motifs.meme.txt")

from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.deep_lift_shap import deep_lift_shap as t_deep_lift_shap
from tangermeme.deep_lift_shap import _nonlinear


def _deep_lift_shap(model, X, args=None, target=0,  batch_size=1024,
	references=dinucleotide_shuffle, n_shuffles=20, return_references=False, 
	hypothetical=False, warning_threshold=0.001, additional_nonlinear_ops=None,
	print_convergence_deltas=False, raw_outputs=False, device='cuda', 
	random_state=None, verbose=False):
	"""A wrapper that registers Chrom/BPNet's custom non-linearities.

	This function is just a wrapper for tangermeme's deep_lift_shap function
	except that it automatically registers the layers that are necessary for
	using BPNet models. Specifically, it registers a scaling that is necessary
	for calculating the profile attributions and also registers the logsumexp
	operation for counts when using the full ChromBPNet model.

	Other than automatically registering the non-linearities, this wrapper does
	not modify the tangermeme outputs or alter the inputs in any way. It is
	simply for convenience so you do not need to reach into bpnet-lite's
	internals each time you want to calculate attributions.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A set of one-hot encoded sequences to calculate attribution values
		for. 

	args: tuple or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Default is None.

	target: int, optional
		The output of the model to calculate gradients/attributions for. This
		will index the last dimension of the predictions. Default is 0.

	batch_size: int, optional
		The number of sequence-reference pairs to pass through DeepLiftShap at
		a time. Importantly, this is not the number of elements in `X` that
		are processed simultaneously (alongside ALL their references) but the
		total number of `X`-`reference` pairs that are processed. This means
		that if you are in a memory-limited setting where you cannot process
		all references for even a single sequence simultaneously that the
		work is broken down into doing only a few references at a time. Default
		is 32.

	references: func or torch.Tensor, optional
		If a function is passed in, this function is applied to each sequence
		with the provided random state and number of shuffles. This function
		should serve to transform a sequence into some form of signal-null
		background, such as by shuffling it. If a torch.Tensor is passed in,
		that tensor must have shape `(len(X), n_shuffles, *X.shape[1:])`, in
		that for each sequence a number of shuffles are provided. Default is
		the function `dinucleotide_shuffle`. 

	n_shuffles: int, optional
		The number of shuffles to use if a function is given for `references`.
		If a torch.Tensor is provided, this number is ignored. Default is 20.

	return_references: bool, optional
		Whether to return the references that were generated during this
		process. Only use if `references` is not a torch.Tensor. Default is 
		False. 

	hypothetical: bool, optional
		Whether to return attributions for all possible characters at each
		position or only for the character that is actually at the sequence.
		Practically, whether to return the returned attributions from captum
		with the one-hot encoded sequence. Default is False.

	warning_threshold: float, optional
		A threshold on the convergence delta that will always raise a warning
		if the delta is larger than it. Normal deltas are in the range of
		1e-6 to 1e-8. Note that convergence deltas are calculated on the
		gradients prior to the aggr_func being applied to them. Default 
		is 0.001. 

	additional_nonlinear_ops: dict or None, optional
		If additional nonlinear ops need to be added to the dictionary of
		operations that can be handled by DeepLIFT/SHAP, pass a dictionary here
		where the keys are class types and the values are the name of the
		function that handle that sort of class. Make sure that the signature
		matches those of `_nonlinear` and `_maxpool` above. This can also be
		used to overwrite the hard-coded operations by passing in a dictionary
		with overlapping key names. If None, do not add any additional 
		operations. Default is None.

	print_convergence_deltas: bool, optional
		Whether to print the convergence deltas for each example when using
		DeepLiftShap. Default is False.

	raw_outputs: bool, optional
		Whether to return the raw outputs from the method -- in this case,
		the multipliers for each example-reference pair -- or the processed
		attribution values. Default is False.

	device: str or torch.device, optional
		The device to move the model and batches to when making predictions. If
		set to 'cuda' without a GPU, this function will crash and must be set
		to 'cpu'. Default is 'cuda'. 

	random_state: int or None or numpy.random.RandomState, optional
		The random seed to use to ensure determinism. If None, the
		process is not deterministic. Default is None. 

	verbose: bool, optional
		Whether to display a progress bar. Default is False.


	Returns
	-------
	attributions: torch.tensor
		If `raw_outputs=False` (default), the attribution values with shape
		equal to `X`. If `raw_outputs=True`, the multipliers for each example-
		reference pair with shape equal to `(X.shape[0], n_shuffles, X.shape[1],
		X.shape[2])`. 

	references: torch.tensor, optional
		The references used for each input sequence, with the shape
		(n_input_sequences, n_shuffles, 4, length). Only returned if
		`return_references = True`. 
	"""

	return t_deep_lift_shap(model=model, X=X, args=args, target=target, 
		batch_size=batch_size, references=references, n_shuffles=n_shuffles, 
		return_references=return_references, hypothetical=hypothetical, 
		warning_threshold=warning_threshold,
		additional_nonlinear_ops={
			_ProfileLogitScaling: _nonlinear,
			_Log: _nonlinear,
			_Exp: _nonlinear
		},
		print_convergence_deltas=print_convergence_deltas, 
		raw_outputs=raw_outputs, device=device, random_state=random_state, 
		verbose=verbose)


def _validate_input(X, name, shape=None, dtype=None, min_value=None,
	max_value=None, ohe=False, ohe_dim=1, allow_N=False):
	"""An internal function for validating properties of the input.

	This function will take in an object and verify characteristics of it, such
	as the type, the datatype of the elements, its shape, etc. If any of these
	characteristics are not met, an error will be raised.


	Parameters
	----------
	X: torch.Tensor
		The object to be verified.

	name: str
		The name to reference the tensor by if an error is raised.

	shape: tuple or None, optional
		The shape the tensor must have. If a -1 is provided at any axis, that
		position is ignored.  If not provided, no check is performed. Default is
		None.

	dtype: torch.dtype or None, optional
		The dtype the tensor must have. If not provided, no check is performed.
		Default is None.

	min_value: float or None, optional
		The minimum value that can be in the tensor, inclusive. If None, no
		check is performed. Default is None.

	max_value: float or None, optional
		The maximum value that can be in the tensor, inclusive. If None, no
		check is performed. Default is None.

	ohe: bool, optional
		Whether the input must be a one-hot encoding, i.e., only consist of
		zeroes and ones. Default is False.

	allow_N: bool, optional
		Whether to allow the return of the character 'N' in the sequence, i.e.
		if pwm at a position is all 0's return N. Default is False.


	Returns
	X: torch.Tensor
		The same object, unmodified, for convenience.
	"""

	if not isinstance(X, torch.Tensor):
		raise ValueError("{} must be a torch.Tensor object".format(name))

	if shape is not None:
		if len(shape) != len(X.shape):
			raise ValueError("{} must have shape {}".format(name, shape))

		for i in range(len(shape)):
			if shape[i] != -1 and shape[i] != X.shape[i]:
				raise ValueError("{} must have shape {}".format(name, shape))


	if dtype is not None and X.dtype != dtype:
		raise ValueError("{} must have dtype {}".format(name, dtype))

	if min_value is not None and X.min() < min_value:
		raise ValueError("{} cannot have a value below {}".format(name,
			min_value))

	if max_value is not None and X.max() > max_value:
		raise ValueError("{} cannot have a value above {}".format(name,
			max_value))

	if ohe:
		values = torch.unique(X)
		if len(values) != 2:
			raise ValueError("{} must be one-hot encoded.".format(name))

		if not all(values == torch.tensor([0, 1], device=X.device)):
			raise ValueError("{} must be one-hot encoded.".format(name))

		if allow_N:
			if not torch.all(torch.sum(X, axis=ohe_dim) <= 1):
				raise ValueError("{} must be one-hot encoded.".format(name) +
					"and contain unknown characters as all-zeroes.")
		else:
			if not torch.all(X.sum(axis=ohe_dim) == 1):
				raise ValueError("{} must be one-hot encoded ".format(name) +
					"and cannot have unknown characters.")				

	return X


def run_modisco_and_shap(
		model, 
		peaks, 
		out_dir, 
		fasta=hg38.fasta, 
		in_window=2114, 
		out_window=1000,
		task='counts', 
		batch_size=64, 
		chrom_sizes=hg38.chrom_sizes,
		sub_sample=None, 
		meme_file=MEME_FILE, 
		max_seqlets=1000_000, 
		width=500, 
		n_control_tracks=0,
		device='cuda',
		debug=False
	):

	print("Modisco output directory:", out_dir)
	out_dir = os.path.join(out_dir, task)
	os.makedirs(out_dir, exist_ok=True)

	if debug:
		sub_sample = 30_000
		max_seqlets = 50_000

	if task == 'profile':
		model = ProfileWrapper(model)
	elif task == 'counts':
		model = CountWrapper(model)
	else:
		raise ValueError(f"Task {task} not recognized. Must be 'profile' or 'counts'")

	# regions_df = pd.read_csv(peaks, sep='\t', header=None)
	regions_df = load_region_df(peaks, chrom_sizes=chrom_sizes)
	if sub_sample is not None and len(regions_df) > sub_sample:
		regions_df = regions_df.sample(sub_sample, random_state=42).reset_index(drop=True)
	print('Number of peaks:', len(regions_df))
	print('Device:', device)

	seq = get_seq(regions_df, pyfaidx.Fasta(fasta), in_window)

	if n_control_tracks > 0:
		args = [torch.zeros(seq.shape[0], n_control_tracks, out_window)]
	else:
		args = None

	if isinstance(seq, np.ndarray):
		seq = torch.tensor(seq.astype(np.float32))
	if seq.shape[-1] == 4:
		seq = seq.permute(0, 2, 1)

	mask = (seq == 0.25).any(dim=(1, 2)) 
	seq = seq[~mask]
	regions_df = regions_df[pd.Series(~mask)]

	## Mask those sequences that are all 0s
	mask = (seq == 0).all(dim=1).any(dim=1)  # Shape: (N, L), True where all 0s in dim=1  
	seq = seq[~mask]  # Filter sequences
	regions_df = regions_df[pd.Series(~mask)]  # Filter regions

	for i in range(seq.shape[0]):
		try:
			X = _validate_input(seq[i], name='seq', shape=(-1, -1), ohe=True, ohe_dim=0)
		except:
			print(i)
			import pdb; pdb.set_trace()

	attr = _deep_lift_shap(model.to(device), seq, batch_size=batch_size, verbose=True, args=args, warning_threshold=1e8, device=device)
	# attr = _captum_deep_lift_shap(model, seq, batch_size=batch_size, verbose=True)
	del model

	shap_dict = generate_shap_dict(seq, attr)
	print('Saving shap dict in h5 format')
	np.object = object
	
	import deepdish
	deepdish.io.save(os.path.join(out_dir, f'shap.h5'), shap_dict, compression='blosc')

	np.save(os.path.join(out_dir, 'attr.npy'), attr)
	np.save(os.path.join(out_dir, 'ohe.npy'), seq)
	regions_df.to_csv(os.path.join(out_dir, 'peaks.bed'), sep='\t', header=False, index=False)
	os.system("sort -k1,1 -k2,2n {} > {}".format(os.path.join(out_dir, 'peaks.bed'), os.path.join(out_dir, 'peaks.sorted.bed')))
	os.system("bgzip -c {} > {}".format(os.path.join(out_dir, 'peaks.sorted.bed'), os.path.join(out_dir, 'peaks.bed.gz')))
	os.system("tabix -p bed {}".format(os.path.join(out_dir, 'peaks.bed.gz')))

	print('Converting shap h5 to bigwig')
	hdf5_to_bigwig(
		os.path.join(out_dir, f'shap.h5'),
		os.path.join(out_dir, 'peaks.bed'),
		chrom_sizes,
		output_prefix=os.path.join(out_dir, f'shap'),
		debug_chr=None,
		tqdm=True
	)

	os.system('modisco motifs -s {} -a {} -n {} -w {} -o {}'.format(
		os.path.join(out_dir, 'ohe.npy'),  
		os.path.join(out_dir, 'attr.npy'), max_seqlets, width, 
		os.path.join(out_dir, f'modisco.h5')
	))

	os.system('modisco report -i {} -o {} -m {}'.format(
		os.path.join(out_dir, 'modisco.h5'),
		os.path.join(out_dir, 'modisco_report'),
		meme_file
	))


	html_to_pdf(
		os.path.join(out_dir, 'modisco_report/motifs.html'),
		os.path.join(out_dir, 'modisco_report.pdf')
	)


def generate_shap_dict(seqs, scores):
	if isinstance(seqs, torch.Tensor):
		seqs = seqs.cpu().numpy()
	if isinstance(scores, torch.Tensor):
		scores = scores.cpu().numpy()

	assert(seqs.shape==scores.shape)
	assert(seqs.shape[1]==4) # one hot encoding, which has bedn transposed

	# construct a dictionary for the raw shap scores and the
	# the projected shap scores
	# MODISCO workflow expects one hot sequences with shape (None,4,inputlen)
	d = {
			'raw': {'seq': seqs.astype(np.int8)},
			'shap': {'seq': scores.astype(np.float16)},
			'projected_shap': {'seq': (seqs*scores).astype(np.float16)}
		}

	return d



	


if __name__ == '__main__':
	import argparse
	import torch
	from .genome import hg38

	parser = argparse.ArgumentParser(description='Run modisco')
	parser.add_argument('--model', type=str, required=True)
	parser.add_argument('--peaks', type=str, required=True)
	parser.add_argument('--out_dir', type=str, required=True)
	parser.add_argument('--task', type=str, default='profile')
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--sub_sample', type=int, default=None)
	parser.add_argument('--chrom_sizes', type=str, default=hg38.chrom_sizes)
	parser.add_argument('--debug', action='store_true')

	args = parser.parse_args()

	from .chrombpnet import BPNet
	if args.model.endswith('.h5'):
		model = BPNet.from_keras(args.model)
	else:
		model = BPNet.load_from_checkpoint(args.model)
		
	if args.task == 'both':
		for task in ['counts', 'profile']:
			run_modisco_and_shap(
				model=model, 
				peaks=args.peaks, 
				out_dir=args.out_dir, 
				task=task,
				batch_size=args.batch_size, 
				sub_sample=args.sub_sample, 
				meme_file=MEME_FILE,
				chrom_sizes=args.chrom_sizes,
				debug=args.debug
			)
	else:
		run_modisco_and_shap(
			model=model, 
			peaks=args.peaks, 
			out_dir=args.out_dir, 
			task=args.task,
			batch_size=args.batch_size, 
			sub_sample=args.sub_sample, 
			meme_file=MEME_FILE,
			chrom_sizes=args.chrom_sizes,
			debug=args.debug
		)