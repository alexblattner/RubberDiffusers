import torch, math
import torch.nn.functional as F
from diffusers.utils import(
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
)
import PIL
import numpy as np
from functools import partial
def find_index(functions,name):
    target_function_index = None
    for index, func in enumerate(functions):
        if (hasattr(func, "__name__") and func.__name__ == name) or (hasattr(func, "func") and hasattr(func.func, "__name__") and func.func.__name__ == name):
            target_function_index = index
            break
    return target_function_index
def apply_dynamicThreasholding(pipe):
    #add defaults
    pipe.denoising_functions.insert(0, partial(dynamicThreasholding_default, pipe))
    #set the DynThresh instance
    new_function_index = find_index(pipe.denoising_functions,"prepare_timesteps")
    pipe.denoising_functions.insert((new_function_index+1), partial(dynamicThreasholding_setdtData, pipe))
    #replace the perform guidance function with a new one
    new_step_function_index=find_index(pipe.denoising_step_functions,"perform_guidance")
    pipe.dynamicThreasholding_stored_perform_guidance=pipe.perform_guidance
    pipe.perform_guidance=partial(perform_guidance,pipe)
    pipe.denoising_step_functions[new_step_function_index]=pipe.perform_guidance
    

    #reverse
    def remover_Correction():
        #undo replacement of the perform guidance function with a new one
        pipe.perform_guidance=pipe.dynamicThreasholding_stored_perform_guidance
        delattr(pipe, f"dynamicThreasholding_stored_perform_guidance")
        pipe.denoising_step_functions[new_step_function_index]=pipe.perform_guidance
        #unset the DynThresh instance
        pipe.denoising_functions.pop((new_function_index+1))
        #remove defaults
        pipe.denoising_functions.pop(0)

    pipe.revert_functions.insert(0,remover_Correction)
def dynamicThreasholding_default(self,**kwargs):
    if kwargs.get('mimic_scale') is None:
        kwargs['mimic_scale']=7.0
    if kwargs.get('threshold_percentile') is None:
        kwargs['threshold_percentile']=1.00
    if kwargs.get('mimic_mode') is None:
        kwargs['mimic_mode']='Constant'
    if kwargs.get('mimic_scale_min') is None:
        kwargs['mimic_scale_min']=0.0
    if kwargs.get('cfg_mode') is None:
        kwargs['cfg_mode']='Constant'
    if kwargs.get('cfg_scale_min') is None:
        kwargs['cfg_scale_min']=0.0
    if kwargs.get('sched_val') is None:
        kwargs['sched_val']=4.0
    if kwargs.get('experiment_mode') is None:
        kwargs['experiment_mode']=0
    if kwargs.get('separate_feature_channels') is None:
        kwargs['separate_feature_channels']=True
    if kwargs.get('scaling_startpoint') is None:
        kwargs['scaling_startpoint']='MEAN'
    if kwargs.get('variability_measure') is None:
        kwargs['variability_measure']='AD'
    if kwargs.get('interpolate_phi') is None:
        kwargs['interpolate_phi']=1.0
    return kwargs
def dynamicThreasholding_setdtData(self, **kwargs):
    mimic_scale=kwargs['mimic_scale']
    threshold_percentile=kwargs['threshold_percentile']
    mimic_mode=kwargs['mimic_mode']
    mimic_scale_min=kwargs['mimic_scale_min']
    cfg_mode=kwargs['mimic_scale']
    cfg_scale_min=kwargs['cfg_scale_min']
    sched_val=kwargs['sched_val']
    experiment_mode=kwargs['experiment_mode']
    separate_feature_channels=kwargs['separate_feature_channels']
    scaling_startpoint=kwargs['scaling_startpoint']
    variability_measure=kwargs['variability_measure']
    interpolate_phi=kwargs['interpolate_phi']
    timesteps=kwargs['timesteps']
    dtData = DynThresh(mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, sched_val, experiment_mode, timesteps.size(0), separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi)
    kwargs['dtData']=dtData
    return kwargs
def perform_guidance(self, i, t, **kwargs):
    do_classifier_free_guidance = kwargs.get('do_classifier_free_guidance')
    guidance_scale = kwargs.get('guidance_scale')
    rescale_noise_cfg = kwargs.get('rescale_noise_cfg')
    noise_pred = kwargs.get('noise_pred')
    dtData = kwargs.get('dtData')
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = dtData.dynthresh(noise_pred, noise_pred_uncond, guidance_scale, None,i)
    kwargs['noise_pred'] = noise_pred
    kwargs['noise_pred_uncond'] = noise_pred_uncond
    return kwargs
######################### DynThresh Core #########################
    
class DynThresh:
    def __init__(self, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, sched_val, experiment_mode, maxSteps, separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi):
        self.mimic_scale = mimic_scale
        self.threshold_percentile = threshold_percentile
        self.mimic_mode = mimic_mode
        self.cfg_mode = cfg_mode
        self.maxSteps = maxSteps
        self.cfg_scale_min = cfg_scale_min
        self.mimic_scale_min = mimic_scale_min
        self.experiment_mode = experiment_mode
        self.sched_val = sched_val
        self.sep_feat_channels = separate_feature_channels
        self.scaling_startpoint = scaling_startpoint
        self.variability_measure = variability_measure
        self.interpolate_phi = interpolate_phi
    def interpretScale(self, scale, mode, min,step):
        scale -= min
        max = self.maxSteps - 1
        if mode == "Constant":
            pass
        elif mode == "Linear Down":
            scale *= 1.0 - (step / max)
        elif mode == "Half Cosine Down":
            scale *= math.cos((step / max))
        elif mode == "Cosine Down":
            scale *= math.cos((step / max) * 1.5707)
        elif mode == "Linear Up":
            scale *= step / max
        elif mode == "Half Cosine Up":
            scale *= 1.0 - math.cos((step / max))
        elif mode == "Cosine Up":
            scale *= 1.0 - math.cos((step / max) * 1.5707)
        elif mode == "Power Up":
            scale *= math.pow(step / max, self.sched_val)
        elif mode == "Power Down":
            scale *= 1.0 - math.pow(step / max, self.sched_val)
        elif mode == "Linear Repeating":
            portion = ((step / max) * self.sched_val) % 1.0
            scale *= (0.5 - portion) * 2 if portion < 0.5 else (portion - 0.5) * 2
        elif mode == "Cosine Repeating":
            scale *= math.cos((step / max) * 6.28318 * self.sched_val) * 0.5 + 0.5
        elif mode == "Sawtooth":
            scale *= ((step / max) * self.sched_val) % 1.0
        scale += min
        return scale

    def dynthresh(self, cond, uncond, cfgScale, weights,step):
        mimicScale = self.interpretScale(self.mimic_scale, self.mimic_mode, self.mimic_scale_min,step)
        cfgScale = self.interpretScale(cfgScale, self.cfg_mode, self.cfg_scale_min,step)
        # uncond shape is (batch, 4, height, width)
        conds_per_batch = cond.shape[0] / uncond.shape[0]
        assert conds_per_batch == int(conds_per_batch), "Expected # of conds per batch to be constant across batches"
        cond_stacked = cond.reshape((-1, int(conds_per_batch)) + uncond.shape[1:])

        ### Normal first part of the CFG Scale logic, basically
        diff = cond_stacked - uncond.unsqueeze(1)
        if weights is not None:
            diff = diff * weights
        relative = diff.sum(1)
        ### Get the normal result for both mimic and normal scale
        mim_target = uncond + relative * mimicScale
        cfg_target = uncond + relative * cfgScale
        ### If we weren't doing mimic scale, we'd just return cfg_target here

        ### Now recenter the values relative to their average rather than absolute, to allow scaling from average
        mim_flattened = mim_target.flatten(2)
        cfg_flattened = cfg_target.flatten(2)
        mim_means = mim_flattened.mean(dim=2).unsqueeze(2)
        cfg_means = cfg_flattened.mean(dim=2).unsqueeze(2)
        mim_centered = mim_flattened - mim_means
        cfg_centered = cfg_flattened - cfg_means

        if self.sep_feat_channels:
            if self.variability_measure == 'STD':
                mim_scaleref = mim_centered.std(dim=2).unsqueeze(2)
                cfg_scaleref = cfg_centered.std(dim=2).unsqueeze(2)
            else: # 'AD'
                mim_scaleref = mim_centered.abs().max(dim=2).values.unsqueeze(2)
                cfg_scaleref = torch.quantile(cfg_centered.abs(), self.threshold_percentile, dim=2).unsqueeze(2)

        else:
            if self.variability_measure == 'STD':
                mim_scaleref = mim_centered.std()
                cfg_scaleref = cfg_centered.std()
            else: # 'AD'
                mim_scaleref = mim_centered.abs().max()
                cfg_scaleref = torch.quantile(cfg_centered.abs(), self.threshold_percentile)

        if self.scaling_startpoint == 'ZERO':
            scaling_factor = mim_scaleref / cfg_scaleref
            result = cfg_flattened * scaling_factor

        else: # 'MEAN'
            if self.variability_measure == 'STD':
                cfg_renormalized = (cfg_centered / cfg_scaleref) * mim_scaleref
            else: # 'AD'
                ### Get the maximum value of all datapoints (with an optional threshold percentile on the uncond)
                max_scaleref = torch.maximum(mim_scaleref, cfg_scaleref)
                ### Clamp to the max
                cfg_clamped = cfg_centered.clamp(-max_scaleref, max_scaleref)
                ### Now shrink from the max to normalize and grow to the mimic scale (instead of the CFG scale)
                cfg_renormalized = (cfg_clamped / max_scaleref) * mim_scaleref

            ### Now add it back onto the averages to get into real scale again and return
            result = cfg_renormalized + cfg_means

        actualRes = result.unflatten(2, mim_target.shape[2:])

        if self.interpolate_phi != 1.0:
            actualRes = actualRes * self.interpolate_phi + cfg_target * (1.0 - self.interpolate_phi)

        if self.experiment_mode == 1:
            num = actualRes.cpu().numpy()
            for y in range(0, 64):
                for x in range (0, 64):
                    if num[0][0][y][x] > 1.0:
                        num[0][1][y][x] *= 0.5
                    if num[0][1][y][x] > 1.0:
                        num[0][1][y][x] *= 0.5
                    if num[0][2][y][x] > 1.5:
                        num[0][2][y][x] *= 0.5
            actualRes = torch.from_numpy(num).to(device=uncond.device)
        elif self.experiment_mode == 2:
            num = actualRes.cpu().numpy()
            for y in range(0, 64):
                for x in range (0, 64):
                    overScale = False
                    for z in range(0, 4):
                        if abs(num[0][z][y][x]) > 1.5:
                            overScale = True
                    if overScale:
                        for z in range(0, 4):
                            num[0][z][y][x] *= 0.7
            actualRes = torch.from_numpy(num).to(device=uncond.device)
        elif self.experiment_mode == 3:
            coefs = torch.tensor([
                #  R       G        B      W
                [0.298,   0.207,  0.208, 0.0], # L1
                [0.187,   0.286,  0.173, 0.0], # L2
                [-0.158,  0.189,  0.264, 0.0], # L3
                [-0.184, -0.271, -0.473, 1.0], # L4
            ], device=uncond.device)
            resRGB = torch.einsum("laxy,ab -> lbxy", actualRes, coefs)
            maxR, maxG, maxB, maxW = resRGB[0][0].max(), resRGB[0][1].max(), resRGB[0][2].max(), resRGB[0][3].max()
            maxRGB = max(maxR, maxG, maxB)
            if step / (self.maxSteps - 1) > 0.2:
                if maxRGB < 2.0 and maxW < 3.0:
                    resRGB /= maxRGB / 2.4
            else:
                if maxRGB > 2.4 and maxW > 3.0:
                    resRGB /= maxRGB / 2.4
            actualRes = torch.einsum("laxy,ab -> lbxy", resRGB, coefs.inverse())

        return actualRes