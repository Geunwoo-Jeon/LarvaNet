from collections import OrderedDict
import numpy as np
import pandas as pd
import torch

# Some modules do the computation themselves using parameters or the parameters of children, treat these as layers
layer_modules = (torch.nn.MultiheadAttention, )

def summary(model, x, *args, layer_modules=layer_modules, print_summary=True, **kwargs):
  """Summarize the given input model.
  Summarized information are 1) output shape, 2) kernel shape,
  3) number of the parameters and 4) operations (Mult-Adds)
  Args:
      model (Module): Model to summarize
      x (Tensor): Input tensor of the model with [N, C, H, W] shape
                  dtype and device have to match to the model
      args, kwargs: Other argument used in `model.forward` function
  """
  def register_hook(module):
    def hook(module, inputs, outputs):
      cls_name = str(module.__class__).split(".")[-1].split("'")[0]
      module_idx = len(summary)

      # Lookup name in a dict that includes parents
      module_name = str(module_idx)
      for name, item in module_names.items():
        if item == module:
          module_name = name
          break
      key = "{}_{}".format(module_idx, name)

      info = OrderedDict()
      info["id"] = id(module)
      if isinstance(outputs, (list, tuple)):
        try:
          info["out"] = list(outputs[0].size())
        except AttributeError:
          # pack_padded_seq and pad_packed_seq store feature into data attribute
          info["out"] = list(outputs[0].data.size())
      else:
        info["out"] = list(outputs.size())

      info["ksize"] = "-"
      info["inner"] = OrderedDict()
      info["params_nt"], info["params"], info["macs"] = 0, 0, 0
      for name, param in module.named_parameters():
        info["params"] += np.int64(param.nelement()) * param.requires_grad
        info["params_nt"] += np.int64(param.nelement()) * (not param.requires_grad)

        if name == "weight":
          ksize = list(param.size())
          # to make [in_shape, out_shape, ksize, ksize]
          if len(ksize) > 1:
            ksize[0], ksize[1] = ksize[1], ksize[0]
          info["ksize"] = ksize

          # ignore N, C when calculate Mult-Adds in ConvNd
          if "Conv" in cls_name:
            info["macs"] += (np.int64(param.nelement()) * np.prod(np.int64(info["out"][2:])))
          else:
            info["macs"] += param.nelement()

        # RNN modules have inner weights such as weight_ih_l0
        elif "weight" in name:
          info["inner"][name] = list(param.size())
          info["macs"] += param.nelement()

      # if the current module is already-used, mark as "(recursive)"
      # check if this module has params
      if list(module.named_parameters()):
        for v in summary.values():
          if info["id"] == v["id"]:
            info["params"] = "(recursive)"

      if info["params"] == 0:
        info["params"], info["macs"] = "-", "-"

      summary[key] = info

    # ignore Sequential and ModuleList and other containers
    if isinstance(module, layer_modules) or not module._modules:
      hooks.append(module.register_forward_hook(hook))

  module_names = get_names_dict(model)

  hooks = []
  summary = OrderedDict()

  model.apply(register_hook)
  try:
    with torch.no_grad():
      model(x) if not (kwargs or args) else model(x, *args, **kwargs)
  except Exception:
    # This can be usefull for debugging
    print("Failed to run torchsummaryX.summary, printing sizes of executed layers:")
    df = pd.DataFrame(summary).T
    print(df)
    raise
  finally:
    for hook in hooks:
      hook.remove()

  # Use pandas to align the columns
  df = pd.DataFrame(summary).T

  df["Mult-Adds"] = pd.to_numeric(df["macs"], errors="coerce")
  df["Params"] = pd.to_numeric(df["params"], errors="coerce")
  df["Non-trainable params"] = pd.to_numeric(df["params_nt"], errors="coerce")
  df = df.rename(columns=dict(
    ksize="Kernel Shape",
    out="Output Shape",
  ))
  df_sum = df.sum()
  df.index.name = "Layer"

  df = df[["Kernel Shape", "Output Shape", "Params", "Mult-Adds"]]
  max_repr_width = max([len(row) for row in df.to_string().split("\n")])

  df_total = pd.DataFrame(
    {
      "Total params": (df_sum["Params"] + df_sum["params_nt"]),
      "Trainable params": df_sum["Params"],
      "Non-trainable params": df_sum["params_nt"],
      "Mult-Adds": df_sum["Mult-Adds"]
    },
    index=['Totals']
  ).T
  
  if print_summary:
    print('')
    print('===== summary =====')
    print('layer_name\tmultiadds\tparams\tparams(non-trainable)\tkernel_shape\toutput_shape')
    macs_list = []
    params_list = []
    params_nt_list = []

    for layer_name, each_summary in summary.items():
      print('%s\t%s\t%s\t%s\t%s\t%s' % (layer_name, each_summary['macs'], each_summary['params'], each_summary['params_nt'], each_summary['ksize'], each_summary['out']))

      if (each_summary['macs'] != "-"):
        macs_list.append(each_summary['macs'])
      if (each_summary['params'] != "-"):
        params_list.append(each_summary['params'])
      if (each_summary['params_nt'] != "-"):
        params_nt_list.append(each_summary['params_nt'])
    
    print('===== ======= =====')

    print('===== total =====')
    print('- multiadds: %s' % (np.sum(macs_list)))
    print('- params: %s' % (np.sum(params_list)))
    print('- params (non-trainable): %s' % (np.sum(params_nt_list)))
    print('===== ===== =====')
    print('')

    option = pd.option_context(
      "display.max_rows", 10000,
      "display.max_columns", 20,
      "display.float_format", pd.io.formats.format.EngFormatter(use_eng_prefix=True)
    )
    with option:
      print("="*max_repr_width)
      print(df.replace(np.nan, "-"))
      print("-"*max_repr_width)
      print(df_total)
      print("="*max_repr_width)
  
  return df, df_total

def get_names_dict(model):
  """Recursive walk to get names including path."""
  names = {}

  def _get_names(module, parent_name=""):
    for key, m in module.named_children():
      cls_name = str(m.__class__).split(".")[-1].split("'")[0]
      num_named_children = len(list(m.named_children()))
      if num_named_children > 0:
        name = parent_name + "." + key if parent_name else key
      else:
        name = parent_name + "." + cls_name + "_" + key if parent_name else key
      names[name] = m

      if isinstance(m, torch.nn.Module):
        _get_names(m, parent_name=name)

  _get_names(model)
  return names