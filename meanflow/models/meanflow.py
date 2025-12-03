import torch

import torch.nn as nn

from models.time_sampler import sample_two_timesteps
from models.ema import init_ema, update_ema_net


class MeanFlow(nn.Module):
    def __init__(self, arch, args, net_configs):
        super(MeanFlow, self).__init__()
        self.net = arch(**net_configs)
        self.args = args

        # Put this in a buffer so that it gets included in the state dict
        self.register_buffer("num_updates", torch.tensor(0))
        
        self.net_ema = init_ema(self.net, arch(**net_configs), args.ema_decay)

        # maintain extra ema nets
        self.ema_decays = args.ema_decays
        for i, ema_decay in enumerate(self.ema_decays):
            self.add_module(f"net_ema{i + 1}", init_ema(self.net, arch(**net_configs), ema_decay))

        # expose current epoch
        self.epoch = 0

        # track fid progression
        self.fid0 = None
        self.fid = None

        # maintain q value
        self.ssize = args.ssize

    def update_ema(self):
        self.num_updates += 1
        # num_updates = self.num_updates.item()
        num_updates = self.num_updates

        update_ema_net(self.net, self.net_ema, num_updates)

        # update extra ema
        for i in range(len(self.ema_decays)):
            update_ema_net(self.net, self._modules[f"net_ema{i + 1}"], num_updates)

    def forward_with_loss(self, x, aug_cond):

        device = x.device
        e = torch.randn_like(x).to(device)
        t, r = sample_two_timesteps(self.args, num_samples=x.shape[0], device=device)
        t, r = t.view(-1, 1, 1, 1), r.view(-1, 1, 1, 1)

        z = (1 - t) * x + t * e
        v = e - x

        # define network function
        def u_func(z, t, r):
            h = t - r
            return self.net(z, (t.view(-1), h.view(-1)), aug_cond)

        dtdt = torch.ones_like(t)
        drdt = torch.zeros_like(r)

        with torch.amp.autocast("cuda", enabled=False):
            u_pred, dudt = torch.func.jvp(u_func, (z, t, r), (v, dtdt, drdt))
        
            u_tgt = (v - (t - r) * dudt).detach()

            loss = (u_pred - u_tgt)**2
            loss = loss.sum(dim=(1, 2, 3))  # squared l2 loss
            
            # adaptive weighting
            adp_wt = (loss.detach() + self.args.norm_eps) ** self.args.norm_p
            loss = loss / adp_wt

            # summary statistics
            loss_det = loss.detach()
            loss_min = loss_det.min().item()
            loss_max = loss_det.max().item()
            loss_mean = loss_det.mean().item()
            loss_std = loss_det.std().item()

            if self.num_updates % 100 == 0:
                print(
                    f"[loss stats] epoch={self.epoch} "
                    f"min={loss_min:.4f} max={loss_max:.4f} "
                    f"mean={loss_mean:.4f} std={loss_std:.4f}"
                )

            ssize = self.ssize
            batch_size = loss.size(0)

            if self.args.method == 0:
                # baseline SGD
                loss = torch.mean(loss)
            else:
                # ordered SGD
                if self.args.method == 1:
                    # constant q
                    pass
                elif self.args.method == 2:
                    # epoch-based q
                    epoch_progress = float(self.epoch + 1) / float(self.args.epochs)
                    if epoch_progress >= 0.875 and ssize > batch_size // 8:
                        ssize = max(1, batch_size // 8)
                    elif epoch_progress >= 0.75 and ssize > batch_size // 4:
                        ssize = max(1, batch_size // 4)
                    elif epoch_progress >= 0.50 and ssize > batch_size // 2:
                        ssize = max(1, batch_size // 2)
                elif self.args.method == 3:
                    # fid-based q
                    if self.fid0 is None:
                        self.fid0 = self.fid
                    else:
                        fid_progress = 100.0 * (self.fid0 - self.fid) / self.fid0
                        if fid_progress >= 99.0 and ssize > batch_size // 16:
                            ssize = max(1, batch_size // 16)
                        elif fid_progress >= 97.0 and ssize > batch_size // 8:
                            ssize = max(1, batch_size // 8)
                        elif fid_progress >= 94.0 and ssize > batch_size // 4:
                            ssize = max(1, batch_size // 4)
                        elif fid_progress >= 90.0 and ssize > batch_size // 2:
                            ssize = max(1, batch_size // 2)

                loss = torch.mean(torch.topk(loss, min(ssize, batch_size), sorted=False, dim=0)[0])
                self.ssize = ssize
        
        return loss
    
    def sample(self, samples_shape, net=None, device=None):
        net = net if net is not None else self.net_ema                

        e = torch.randn(samples_shape, dtype=torch.float32, device=device)
        z_1 = e
        t = torch.ones(samples_shape[0], device=device)
        r = torch.zeros(samples_shape[0], device=device)
        u = net(z_1, (t, t - r), aug_cond=None)
        z_0 = z_1 - u
        return z_0
