import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import numpy as np

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .wavelet_util import take_wavelet

# INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        initial_lg_loss_scale,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0
        
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 1
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            initial_lg_loss_scale = initial_lg_loss_scale
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]
            
        self.output_model_stastics()

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def output_model_stastics(self):
        num_params_total = sum(p.numel() for p in self.model.parameters())
        num_params_train = 0
        num_params_pre_load = 0
        
        for param_group in self.opt.param_groups:
            if param_group['lr'] >0:
                num_params_train += sum(p.numel() for p in param_group['params'] if p.requires_grad==True)
    
        if hasattr(self, 'pre_load_params'):
            num_params_pre_load=sum(p.numel() for name, p in self.model.named_parameters() if name in self.pre_load_params)
            #[p.mean().item() for _, p in self.model.named_parameters()]
        if num_params_total > 1e6:
            num_params_total /= 1e6
            num_params_train /= 1e6
            num_params_pre_load /= 1e6
            params_total_label = 'M'
        elif num_params_total > 1e3:
            num_params_total /= 1e3
            num_params_train /= 1e3
            num_params_pre_load = 1e3
            params_total_label = 'k'

        logger.log("Total Parameters:{:.2f}{}".format(num_params_total, params_total_label))
        logger.log("Total Training Parameters:{:.2f}{}".format(num_params_train, params_total_label))
        logger.log("Total Loaded Parameters:{:.2f}{}".format(num_params_pre_load, params_total_label))
        if dist.get_rank() == 0:
            print("Total Parameters:{:.2f}{}".format(num_params_total, params_total_label))
            print("Total Training Parameters:{:.2f}{}".format(num_params_train, params_total_label))
            print("Total Loaded Parameters:{:.2f}{}".format(num_params_pre_load, params_total_label))

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            state_dict = dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
            logger.log(f"loading model from checkpoint: {self.resume_checkpoint}...")
            self.pre_load_params = state_dict.keys()
            self.model.load_state_dict(state_dict)

        dist_util.sync_params(self.model.parameters())
        print("loading model params finished")

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            print("optimizer path exists")
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )

            self.opt.load_state_dict(state_dict)
            print("finish loading optimizer")

    def run_loop(self):

        running_loss_ll = 0.0
        running_loss_lh = 0.0
        running_loss_hl = 0.0

        running_grads_norm_ll = 0.0
        running_grads_norm_lh = 0.0
        running_grads_norm_hl = 0.0
        running_others_grad_norm = 0.0



        running_params_norm_ll = 0.0
        running_params_norm_lh = 0.0
        running_params_norm_hl = 0.0
        running_others_param_norm = 0.0

        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            #if self.step==self.stop_iter:
            #    break
            batch, cond = next(self.data)
            batch = take_wavelet(batch)
            losses = self.run_step(batch, cond)

            running_loss_ll = (running_loss_ll * ((self.step -1) % self.log_interval) + losses['mse_ll'].mean().item()) / ((self.step-1)  % self.log_interval + 1)
            running_loss_lh = (running_loss_lh * ((self.step -1) % self.log_interval) + losses['mse_lh'].mean().item()) / ((self.step-1)  % self.log_interval + 1)
            running_loss_hl = (running_loss_hl * ((self.step -1) % self.log_interval) + losses['mse_hl'].mean().item()) / ((self.step-1)  % self.log_interval + 1)

            norms_dict = self.compute_statistics_norms(self.ddp_model, self.mp_trainer.lg_loss_scale)
            
            running_grads_norm_ll = (running_grads_norm_ll * ((self.step-1)  % self.log_interval) + norms_dict['grads_ll']) / ((self.step-1)  % self.log_interval + 1)
            running_grads_norm_lh = (running_grads_norm_lh * ((self.step-1)  % self.log_interval) + norms_dict['grads_lh']) / ((self.step-1)  % self.log_interval + 1)
            running_grads_norm_hl = (running_grads_norm_hl * ((self.step-1)  % self.log_interval) + norms_dict['grads_hl']) / ((self.step-1)  % self.log_interval + 1)
            running_others_grad_norm = (running_others_grad_norm * ((self.step-1)  % self.log_interval) + norms_dict['grads_others']) / ((self.step-1)  % self.log_interval + 1)



            running_params_norm_ll = (running_params_norm_ll * ((self.step-1)  % self.log_interval) + norms_dict['params_ll']) / ((self.step-1)  % self.log_interval + 1)
            running_params_norm_lh = (running_params_norm_lh * ((self.step-1)  % self.log_interval) + norms_dict['params_lh']) / ((self.step-1)  % self.log_interval + 1)
            running_params_norm_hl = (running_params_norm_hl * ((self.step-1)  % self.log_interval) + norms_dict['params_hl']) / ((self.step-1)  % self.log_interval + 1)
            running_others_param_norm = (running_others_param_norm * ((self.step-1)  % self.log_interval) + norms_dict['params_others']) / ((self.step-1)  % self.log_interval + 1)


            running_total_loss = running_loss_ll + running_loss_lh + running_loss_hl
            running_total_grads_norm = running_grads_norm_ll + running_grads_norm_lh + running_grads_norm_hl + running_others_grad_norm
            running_total_params_norm = running_params_norm_ll + running_params_norm_lh + running_params_norm_hl + running_others_param_norm


            if self.step % self.log_interval == 0:
                if dist.get_rank() == 0:
                    print(f"step:{(self.step + self.resume_step):06d} | total_loss: {running_total_loss:.4f} | ll_loss: {running_loss_ll:.4f} | lh_loss: {running_loss_lh:.4f} | hl_loss: {running_loss_hl:.4f}") # assume weights are all ones!
            
                logger.dumpkvs()

            if self.step % self.log_interval == 0:
                for logger_format in logger.get_current().output_formats:
                    if str(logger_format).split()[0].split(".")[-1] == "TensorBoardOutputFormat": #str(logger_format).split(".")[-1].startswith("TensorBoardOutputFormat")
                        logger_format.tb_step = self.step + self.resume_step
                        logger_format.writekvs({"loss": {"total": running_total_loss,
                                                    "ll": running_loss_ll,
                                                    "lh": running_loss_lh,
                                                    "hl": running_loss_hl}}) 
                        
                        logger_format.writekvs({"grads_norm": {"total": running_total_grads_norm,
                                                                "ll": running_grads_norm_ll,
                                                                "lh": running_grads_norm_lh,
                                                                "hl": running_grads_norm_hl,
                                                                "others": running_others_grad_norm}})
                        

                        logger_format.writekvs({"params_norm": {"total": running_total_params_norm,
                                                                "ll": running_params_norm_ll,
                                                                "lh": running_params_norm_lh,
                                                                "hl": running_params_norm_hl,
                                                                "othres": running_others_param_norm}})
                        
                        running_total_loss = 0.0
                        running_loss_ll = 0.0
                        running_loss_lh = 0.0
                        running_loss_hl = 0.0


                        running_total_grads_norm = 0.0
                        running_grads_norm_ll = 0.0
                        running_grads_norm_lh = 0.0
                        running_grads_norm_hl = 0.0
                        running_others_grad_norm = 0.0

                        running_total_params_norm = 0.0
                        running_params_norm_ll = 0.0
                        running_params_norm_lh = 0.0
                        running_params_norm_hl = 0.0
                        running_others_param_norm = 0.0



            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step-1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        losses = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return losses

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch["ll"].shape[0], self.microbatch):
            micro = {}
            for subband, subband_val in batch.items():
                micro[subband] = subband_val[i : i + self.microbatch].to(dist_util.dev())

            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch["ll"].shape[0]
            t, weights = self.schedule_sampler.sample(micro["ll"].shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

            return losses

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def compute_statistics_norms(self,model, loss_scale):

        ll_total_params_norm = 0.0
        lh_total_params_norm = 0.0
        hl_total_params_norm = 0.0
        total_others_params_norm = 0.0

        ll_total_grads_norm = 0.0
        lh_total_grads_norm = 0.0
        hl_total_grads_norm = 0.0
        total_others_grads_norm = 0.0


        for name, param in model.named_parameters():
            with th.no_grad():
                if "ll" in str(name) and param.requires_grad:
                    ll_total_params_norm += th.norm(param, p=2, dtype=th.float32).item() ** 2
                    if param.grad is not None:
                        ll_total_grads_norm += th.norm(param.grad, p=2, dtype=th.float32).item() ** 2

                elif "lh" in str(name) and param.requires_grad:
                    lh_total_params_norm += th.norm(param, p=2, dtype=th.float32).item() ** 2
                    if param.grad is not None:
                        lh_total_grads_norm += th.norm(param.grad, p=2, dtype=th.float32).item() ** 2


                elif "hl" in str(name) and param.requires_grad:   
                    hl_total_params_norm += th.norm(param, p=2, dtype=th.float32).item() ** 2
                    if param.grad is not None:
                        hl_total_grads_norm += th.norm(param.grad, p=2, dtype=th.float32).item() ** 2

                else:
                    total_others_params_norm += th.norm(param, p=2, dtype=th.float32).item() ** 2
                    if param.grad is not None:
                        total_others_grads_norm += th.norm(param.grad, p=2, dtype=th.float32).item() ** 2

        scaling_fac = 2 ** loss_scale if self.use_fp16 else 1.0

        return {"grads_ll": np.sqrt(ll_total_grads_norm) / scaling_fac,
                "grads_lh": np.sqrt(lh_total_grads_norm) / scaling_fac,
                "grads_hl": np.sqrt(hl_total_grads_norm) / scaling_fac,
                "grads_others": np.sqrt(total_others_grads_norm) / scaling_fac,
                "params_ll": np.sqrt(ll_total_params_norm),
                "params_lh": np.sqrt(lh_total_params_norm),
                "params_hl": np.sqrt(hl_total_params_norm),
                "params_others": np.sqrt(total_others_params_norm)}


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
