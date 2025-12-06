# Author: Lei Xiong <jsxlei@gmail.com>

import torch
import torch.nn as nn

from chrombpnet.bpnet import BPNet


class _Exp(nn.Module):
	def __init__(self):
		super(_Exp, self).__init__()

	def forward(self, X):
		return torch.exp(X)


class _Log(nn.Module):
	def __init__(self):
		super(_Log, self).__init__()

	def forward(self, X):
		return torch.log(X)
     

def _to_numpy(tensor):
    return tensor.detach().cpu().numpy()

# adapt from BPNet in bpnet-lite, credit goes to Jacob Schreiber <jmschreiber91@gmail.com>
class ChromBPNet(nn.Module):
    """A ChromBPNet model.

    ChromBPNet is an extension of BPNet to handle chromatin accessibility data,
    in contrast to the protein binding data that BPNet handles. The distinction
    between these data types is that an enzyme used in DNase-seq and ATAC-seq
    experiments itself has a soft sequence preference, meaning that the
    strength of the signal is driven by real biology but that the exact read
    mapping locations are driven by the soft sequence bias of the enzyme.

    ChromBPNet handles this by treating the data using two models: a bias
    model that is initially trained on background (non-peak) regions where
    the bias dominates, and an accessibility model that is subsequently trained
    using a frozen version of the bias model. The bias model learns to remove
    the enzyme bias so that the accessibility model can learn real motifs.


    Parameters
    ----------
    bias: torch.nn.Module 
        This model takes in sequence and outputs the shape one would expect in 
        ATAC-seq data due to Tn5 bias alone. This is usually a BPNet model
        from the bpnet-lite repo that has been trained on GC-matched non-peak
        regions.

    accessibility: torch.nn.Module
        This model takes in sequence and outputs the accessibility one would 
        expect due to the components of the sequence, but also takes in a cell 
        representation which modifies the parameters of the model, hence, 
        "dynamic." This model is usually a DynamicBPNet model, defined below.

    name: str
        The name to prepend when saving the file.
    """

    def __init__(self, 
        config,
        **kwargs
        ):
        super().__init__()

        self.chrombpnet_wo_bias = BPNet(        
            out_dim=config.out_dim,
            n_filters=config.n_filters, 
            n_layers=config.n_layers, 
            conv1_kernel_size=config.conv1_kernel_size,
            profile_kernel_size=config.profile_kernel_size,
            n_outputs=config.n_outputs, 
            n_control_tracks=config.n_control_tracks, 
            profile_output_bias=config.profile_output_bias, 
            count_output_bias=config.count_output_bias, 
        )

        self.bias = BPNet(out_dim=config.out_dim, n_layers=4, n_filters=128)

        self._log = _Log()
        self._exp1 = _Exp()
        self._exp2 = _Exp()
		
        self.n_control_tracks = config.n_control_tracks

        self.tf_style_reinit()

    def tf_style_reinit(self):
        """
        Re-initializes model weights for Linear and Conv1d layers using
        TensorFlow's default: Xavier/Glorot uniform for weights, zeros for bias.
        Operates in-place!
        """
        # print("Reinitializing with TF strategy")
        for m in self.chrombpnet_wo_bias.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, **kwargs):
        """A forward pass through the network.

        This function is usually accessed through calling the model, e.g.
        doing `model(x)`. The method defines how inputs are transformed into
        the outputs through interactions with each of the layers.


        Parameters
        ----------
        x: torch.tensor, shape=(-1, 4, 2114)
            A one-hot encoded sequence tensor.

        X_ctl: ignore
            An ignored parameter for consistency with attribution functions.


        Returns
        -------
        y_profile: torch.tensor, shape=(-1, 1000)
            The predicted logit profile for each example. Note that this is not
            a normalized value.
        """
        acc_profile, acc_counts = self.chrombpnet_wo_bias(x)
        bias_profile, bias_counts = self.bias(x)

        y_profile = acc_profile + bias_profile
        y_counts = self._log(self._exp1(acc_counts) + self._exp2(bias_counts))
        
        # DO NOT SQUEEZE y_counts, as it is needed for running deep_lift_shap
        return y_profile.squeeze(1), y_counts #.squeeze() 