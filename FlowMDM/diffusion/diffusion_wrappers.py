
import torch
import torch as th
from copy import deepcopy


class DiffusionWrapper_FlowMDM():
    def __init__(self, args, diffusion, model):
        self.model = model
        self.diffusion = diffusion
        self.guidance_param = args.guidance_param

    def add_bias_and_absolute_matrices(self, model_kwargs, shape, device):
        """
        We build:
        > pe_bias --> [T, T] matrix with -inf and 0's limiting where the attention during APE mode focuses (0's), i.e., inside each subsequence
        > pos_pe_abs --> [T] matrix with the absolute position of each frame in each subsequence (for injecting the APE sinusoidal correctly during APE mode).
        """
        nframes = shape[-1]

        pos_pe_abs = torch.zeros((nframes, ), device=device, dtype=torch.float32)
        pe_bias = torch.full((nframes, nframes), float('-inf'), device=device, dtype=torch.float32)

        s = 0 # start
        for length in model_kwargs['y']['lengths']:
            pos_pe_abs[s:s+length] = torch.arange(length, device=device, dtype=torch.float32)
            pe_bias[s:s+length, s:s+length] = 0 # only attend to the segment for the absolute modeling part of the schedule
            s += length

        model_kwargs['y']['pe_bias'] = pe_bias # in FlowMDM forward, it is selected according to the BPE schedule if active
        model_kwargs['y']['pos_pe_abs'] = pos_pe_abs.unsqueeze(0) # needs batch size

    def add_conditions_mask(self, model_kwargs, num_frames, device):
        """
        We build a mask of shape [S, T, 1] where S is the number of motion subsequences, T is the max. sequence length.
        For each subsequence, the mask is True only for the frames that belong to the subsequence.
        """
        num_samples = len(model_kwargs["y"]["lengths"])
        conditions_mask = th.zeros((num_samples, num_frames, 1), device=device, dtype=th.bool)
        s = 0
        MARGIN = 0
        for i, length in enumerate(model_kwargs["y"]["lengths"]):
            conditions_mask[i, s+MARGIN:s+length-MARGIN, :] = True # all batch elements have the same instructions
            s += length
        model_kwargs['y']['conditions_mask'] = conditions_mask

    def p_sample_loop(
        self,
        model_kwargs=None, # list of dicts
        **kwargs,
    ):
        final = None
        for i, sample in enumerate(self.p_sample_loop_progressive(
            model_kwargs=model_kwargs,
            **kwargs,
        )):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        noise=None,
        model_kwargs=None, # list of dicts
        device=None,
        progress=False,
        **kwargs,
    ):
        bs, nframes = 1, model_kwargs['y']['lengths'].sum().item()
        shape = (bs, self.model.njoints, self.model.nfeats, nframes) # all batch elements form the same sequence

        if device is None:
            device = next(self.model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        model_kwargs = deepcopy(model_kwargs)
        self.add_conditions_mask(model_kwargs, nframes, device)
        self.add_bias_and_absolute_matrices(model_kwargs, shape, device)
        model_kwargs["y"]["mask"] = th.ones((bs, nframes), device=device, dtype=th.bool)
        model_kwargs["y"]["lengths"] = th.tensor([nframes], device=device, dtype=th.int64)
        model_kwargs["y"]["scale"] = th.ones(bs, device=device) * self.guidance_param
        # texts are joined as well
        model_kwargs["y"]["all_texts"] = [model_kwargs["y"]["text"], ]
        model_kwargs["y"]["all_lengths"] = [model_kwargs["y"]["lengths"], ]
        model_kwargs["y"]["text"] = " -- ".join(model_kwargs["y"]["text"])

        indices = list(range(self.diffusion.num_timesteps))[::-1]
        
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for t in indices:
                
            with th.no_grad():

                t = th.tensor([t] * shape[0], device=device)
                out = self.diffusion.p_sample(
                    self.model,
                    img,
                    t,
                    model_kwargs=model_kwargs,
                    **kwargs,
                )

                yield out
                img = out["sample"]

    def ddim_sample_loop(
        self,
            model_kwargs=None,  # list of dicts
            **kwargs,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        #
        # final = None
        # for sample in self.ddim_sample_loop_progressive(
        #     model,
        #     shape,
        #     noise=noise,
        #     clip_denoised=clip_denoised,
        #     denoised_fn=denoised_fn,
        #     cond_fn=cond_fn,
        #     model_kwargs=model_kwargs,
        #     device=device,
        #     progress=progress,
        #     eta=eta,
        #     skip_timesteps=skip_timesteps,
        #     init_image=init_image,
        #     randomize_class=randomize_class,
        #     cond_fn_with_grad=cond_fn_with_grad,
        # ):
        #     final = sample
        # return final["sample"]

        final = None
        for i, sample in enumerate(self.ddim_sample_loop_progressive(
                model_kwargs=model_kwargs,
                **kwargs,
        )):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
            noise=None,
            model_kwargs=None,  # list of dicts
            device=None,
            progress=False,
            **kwargs,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        # if device is None:
        #     device = next(model.parameters()).device
        # assert isinstance(shape, (tuple, list))
        # if noise is not None:
        #     img = noise
        # else:
        #     img = th.randn(*shape, device=device)
        #
        # if skip_timesteps and init_image is None:
        #     init_image = th.zeros_like(img)
        #
        # indices = list(range(self.num_timesteps - skip_timesteps))[::-1]
        #
        # if init_image is not None:
        #     my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
        #     img = self.q_sample(init_image, my_t, img)
        #
        # if progress:
        #     # Lazy import so that we don't depend on tqdm.
        #     from tqdm.auto import tqdm
        #
        #     indices = tqdm(indices)
        #
        # for i in indices:
        #     t = th.tensor([i] * shape[0], device=device)
        #     if randomize_class and 'y' in model_kwargs:
        #         model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
        #                                        size=model_kwargs['y'].shape,
        #                                        device=model_kwargs['y'].device)
        #     with th.no_grad():
        #         sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
        #         out = sample_fn(
        #             model,
        #             img,
        #             t,
        #             clip_denoised=clip_denoised,
        #             denoised_fn=denoised_fn,
        #             cond_fn=cond_fn,
        #             model_kwargs=model_kwargs,
        #             eta=eta,
        #         )
        #         yield out
        #         img = out["sample"]

        bs, nframes = 1, model_kwargs['y']['lengths'].sum().item()
        shape = (bs, self.model.njoints, self.model.nfeats, nframes)  # all batch elements form the same sequence

        if device is None:
            device = next(self.model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        model_kwargs = deepcopy(model_kwargs)
        self.add_conditions_mask(model_kwargs, nframes, device)
        self.add_bias_and_absolute_matrices(model_kwargs, shape, device)
        model_kwargs["y"]["mask"] = th.ones((bs, nframes), device=device, dtype=th.bool)
        model_kwargs["y"]["lengths"] = th.tensor([nframes], device=device, dtype=th.int64)
        model_kwargs["y"]["scale"] = th.ones(bs, device=device) * self.guidance_param
        # texts are joined as well
        model_kwargs["y"]["all_texts"] = [model_kwargs["y"]["text"], ]
        model_kwargs["y"]["all_lengths"] = [model_kwargs["y"]["lengths"], ]
        model_kwargs["y"]["text"] = " -- ".join(model_kwargs["y"]["text"])

        indices = list(range(self.diffusion.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for t in indices:
            with th.no_grad():
                t = th.tensor([t] * shape[0], device=device)
                out = self.diffusion.ddim_sample(
                    self.model,
                    img,
                    t,
                    model_kwargs=model_kwargs,
                    **kwargs,
                )

                yield out
                img = out["sample"]

    def ddim_sample_loop_full_chain(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            skip_timesteps=0,
            init_image=None,
            randomize_class=False,
            cond_fn_with_grad=False,
    ):
        """
        Generate samples from the model using DDIM while keeping the full gradient chain.

        Same usage as p_sample_loop().
        """
        # make this works with non-grad
        # assert noise.requires_grad == True

        final = None
        for i, sample in enumerate(
                self.ddim_sample_loop_full_chain_progressive(
                    model,
                    shape,
                    noise=noise,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=progress,
                    eta=eta,
                    skip_timesteps=skip_timesteps,
                    init_image=init_image,
                    randomize_class=randomize_class,
                    cond_fn_with_grad=cond_fn_with_grad,
                )):
            # final = sample
            # if dump_steps is not None and i in dump_steps:
            #     # dump.append(deepcopy(sample["sample"]))
            #     dump.append(deepcopy(sample["pred_xstart"]))
            final = sample
        # if dump_steps is not None:
        #     return dump

        return final["sample"]

    def ddim_sample_loop_full_chain_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            skip_timesteps=0,
            init_image=None,
            randomize_class=False,
            cond_fn_with_grad=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device,
                           dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        # NOTE: for debugging
        previous_xstart = img

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            # sample_fn = self.ddim_sample_with_grad_chain
            out = self.ddim_sample_with_grad_chain(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                eta=eta,
                previous_xstart=previous_xstart  # NOTE: for debugging
            )
            yield out
            img = out["sample"]
            previous_xstart = out["pred_xstart"]

    def ddim_sample_with_grad_chain(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
            previous_xstart=None,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        with th.enable_grad():
            # x = x.detach().requires_grad_()
            out_orig = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                previous_xstart=previous_xstart,
            )
            # if cond_fn is not None:
            #     out = self.condition_score_with_grad(cond_fn,
            #                                          out_orig,
            #                                          x,
            #                                          t,
            #                                          model_kwargs=model_kwargs)
            # else:
            #     out = out_orig
            out = out_orig

        # out["pred_xstart"] = out["pred_xstart"].detach()

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t,
                                              x.shape)
        sigma = (eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) *
                 th.sqrt(1 - alpha_bar / alpha_bar_prev))
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (out["pred_xstart"] * th.sqrt(alpha_bar_prev) +
                     th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {
            "sample": sample,
            "pred_xstart": out_orig["pred_xstart"].detach()
        }