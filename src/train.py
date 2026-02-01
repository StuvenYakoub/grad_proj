from models.generator import TSCNet
from models import discriminator
import os
from data import dataloader
import torch.nn.functional as F
import torch
from utils import power_compress, power_uncompress
import logging
from torchinfo import summary
import argparse

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--log_interval", type=int, default=1)
parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=16000*2, help="cut length, default is 2 seconds in denoise "
                                                                 "and dereverberation")
parser.add_argument("--data_dir", type=str, default='../VCTK-DEMAND',
                    help="../")
parser.add_argument("--save_model_dir", type=str, default='./saved_model_normal',
                    help="dir of saved model")
parser.add_argument("--loss_weights", type=list, default=[0.1, 0.9, 0.2, 0.05],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
parser.add_argument("--resume", type=str, default=None, help="path to a full checkpoint .pt to resume from")
parser.add_argument("--start_epoch", type=int, default=0, help="manually set starting epoch (optional)")

args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
    def __init__(self, train_ds, test_ds, gpu_id: int):
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.model = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).cuda()
        summary(
            self.model, [(1, 2, args.cut_len // self.hop + 1, int(self.n_fft / 2) + 1)]
        )
        self.discriminator = discriminator.Discriminator(ndf=16).cuda()
        summary(
            self.discriminator,
            [
                (1, 1, int(self.n_fft / 2) + 1, args.cut_len // self.hop + 1),
                (1, 1, int(self.n_fft / 2) + 1, args.cut_len // self.hop + 1),
            ],
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr)
        self.optimizer_disc = torch.optim.AdamW(
            self.discriminator.parameters(), lr=2 * args.init_lr
        )

        self.model = DDP(self.model, device_ids=[gpu_id])
        self.discriminator = DDP(self.discriminator, device_ids=[gpu_id])
        self.gpu_id = gpu_id

    def load_checkpoint(self, ckpt_path, scheduler_G=None, scheduler_D=None):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        self.model.module.load_state_dict(ckpt["model"])
        self.discriminator.module.load_state_dict(ckpt["discriminator"])

        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.optimizer_disc.load_state_dict(ckpt["optimizer_disc"])

        device = torch.device(f"cuda:{self.gpu_id}")
        self._optimizer_to_device(self.optimizer, device)
        self._optimizer_to_device(self.optimizer_disc, device)

        if scheduler_G is not None and "scheduler_G" in ckpt:
            scheduler_G.load_state_dict(ckpt["scheduler_G"])
        if scheduler_D is not None and "scheduler_D" in ckpt:
            scheduler_D.load_state_dict(ckpt["scheduler_D"])

        return ckpt.get("epoch", -1)
    
    def _optimizer_to_device(self, optimizer, device):
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    def forward_generator_step(self, clean, noisy):

        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(
            clean * c, 0, 1
        )

        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
            return_complex=False,
        )
        clean_spec = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
            return_complex=False,
        )
        if torch.is_complex(noisy_spec):
            noisy_spec = torch.view_as_real(noisy_spec)  # -> (..., 2)
        if torch.is_complex(clean_spec):
            clean_spec = torch.view_as_real(clean_spec)  # -> (..., 2)

        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        est_real, est_imag = self.model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

        est_spec_ri = power_uncompress(est_real, est_imag).squeeze(1)  # expected: [B, F, T, 2]
        est_spec_complex = torch.view_as_complex(est_spec_ri.contiguous())

        est_audio = torch.istft(
            est_spec_complex,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )

        return {
            "est_real": est_real,
            "est_imag": est_imag,
            "est_mag": est_mag,
            "clean_real": clean_real,
            "clean_imag": clean_imag,
            "clean_mag": clean_mag,
            "est_audio": est_audio,
        }

    def calculate_generator_loss(self, generator_outputs):

        predict_fake_metric = self.discriminator(
            generator_outputs["clean_mag"], generator_outputs["est_mag"]
        )
        gen_loss_GAN = F.mse_loss(
            predict_fake_metric.flatten(), generator_outputs["one_labels"].float()
        )

        loss_mag = F.mse_loss(
            generator_outputs["est_mag"], generator_outputs["clean_mag"]
        )
        loss_ri = F.mse_loss(
            generator_outputs["est_real"], generator_outputs["clean_real"]
        ) + F.mse_loss(generator_outputs["est_imag"], generator_outputs["clean_imag"])

        time_loss = torch.mean(
            torch.abs(generator_outputs["est_audio"] - generator_outputs["clean"])
        )

        loss = (
            args.loss_weights[0] * loss_ri
            + args.loss_weights[1] * loss_mag
            + args.loss_weights[2] * time_loss
            + args.loss_weights[3] * gen_loss_GAN
        )

        return loss

    def calculate_discriminator_loss(self, generator_outputs):

        length = generator_outputs["est_audio"].size(-1)
        est_audio_list = list(generator_outputs["est_audio"].detach().cpu().numpy())
        clean_audio_list = list(generator_outputs["clean"].cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)

        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None:
            predict_enhance_metric = self.discriminator(
                generator_outputs["clean_mag"], generator_outputs["est_mag"].detach()
            )
            predict_max_metric = self.discriminator(
                generator_outputs["clean_mag"], generator_outputs["clean_mag"]
            )
            discrim_loss_metric = F.mse_loss(
                predict_max_metric.flatten(), generator_outputs["one_labels"]
            ) + F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
        else:
            discrim_loss_metric = None

        return discrim_loss_metric

    def train_step(self, batch):

        # Trainer generator
        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        one_labels = torch.ones(args.batch_size).to(self.gpu_id)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        loss = self.calculate_generator_loss(generator_outputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Train Discriminator
        discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)

        if discrim_loss_metric is not None:
            self.optimizer_disc.zero_grad()
            discrim_loss_metric.backward()
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.0])

        return loss.item(), discrim_loss_metric.item()

    @torch.no_grad()
    def test_step(self, batch):

        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        one_labels = torch.ones(args.batch_size).to(self.gpu_id)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        loss = self.calculate_generator_loss(generator_outputs)

        discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)
        if discrim_loss_metric is None:
            discrim_loss_metric = torch.tensor([0.0])

        return loss.item(), discrim_loss_metric.item()

    def test(self):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.0
        disc_loss_total = 0.0
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss = self.test_step(batch)
            gen_loss_total += loss
            disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step

        template = "GPU: {}, Generator loss: {}, Discriminator loss: {}"
        logging.info(template.format(self.gpu_id, gen_loss_avg, disc_loss_avg))

        return gen_loss_avg

    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.decay_epoch, gamma=0.5
        )
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_disc, step_size=args.decay_epoch, gamma=0.5
        )

        start_epoch = args.start_epoch

        if args.resume is not None:
            last_epoch = self.load_checkpoint(args.resume, scheduler_G, scheduler_D)
            start_epoch = last_epoch + 1  # continue from next epoch
            if self.gpu_id == 0:
                logging.info(f"Resumed from {args.resume} at epoch {last_epoch}, starting epoch {start_epoch}")

        for epoch in range(start_epoch, args.epochs):
            self.model.train()
            self.discriminator.train()
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                loss, disc_loss = self.train_step(batch)
                template = "GPU: {}, Epoch {}, Step {}, loss: {}, disc_loss: {}"
                if (step % args.log_interval) == 0:
                    logging.info(
                        template.format(self.gpu_id, epoch, step, loss, disc_loss)
                    )
            gen_loss = self.test()
            path = os.path.join(
                args.save_model_dir,
                "CMGAN_epoch_" + str(epoch) + "_" + str(gen_loss)[:5],
            )
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            if self.gpu_id == 0:
                os.makedirs(args.save_model_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_model_dir, f"ckpt_epoch_{epoch}.pt")

                ckpt = {
                    "epoch": epoch,
                    "model": self.model.module.state_dict(),
                    "discriminator": self.discriminator.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "optimizer_disc": self.optimizer_disc.state_dict(),
                    "scheduler_G": scheduler_G.state_dict(),
                    "scheduler_D": scheduler_D.state_dict(),
                }
                torch.save(ckpt, ckpt_path)
            scheduler_G.step()
            scheduler_D.step()


def main(rank: int, world_size: int, args):
    ddp_setup(rank, world_size)
    if rank == 0:
        print(args)
        available_gpus = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        print(available_gpus)
    train_ds, test_ds = dataloader.load_data(
        args.data_dir, args.batch_size, 2, args.cut_len
    )
    trainer = Trainer(train_ds, test_ds, rank)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
