import torch
from typing import Tuple
from dicee.abstracts import AbstractTrainer
import time
import os
import psutil
from tqdm import tqdm

class TorchTrainer(AbstractTrainer):
    """
        TorchTrainer for using single GPU or multi CPUs on a single node

        Arguments
       ----------
       args: ?

       callbacks: list of Abstract callback instances

   """

    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)
        self.loss_function = None
        self.optimizer = None
        self.model = None
        self.train_dataloaders = None
        self.training_step = None
        torch.manual_seed(self.attributes.random_seed)
        torch.cuda.manual_seed_all(self.attributes.random_seed)
        if hasattr(self.attributes,"gpus") and self.attributes.gpus and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.attributes.gpus}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        
        # https://psutil.readthedocs.io/en/latest/#psutil.Process
        self.process = psutil.Process(os.getpid())

    def _run_batch(self, i: int, x_batch, y_batch) -> float:
        """
            Forward anc Backward according to a mini-batch

            Arguments
           ----------
           i : index of a batch
           x_batch: torch.Tensor on selected device
           y_batch: torch.Tensor on selected device
           Returns
           -------
           batch loss (float)
       """
        if self.attributes.gradient_accumulation_steps > 1:
            # (1) Update parameters every gradient_accumulation_steps mini-batch.
            if i % self.attributes.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)
        else:
            # (2) Do not accumulate gradient, zero the gradients per batch.
            self.optimizer.zero_grad(set_to_none=True)
        # (3) Loss Forward and Backward w.r.t the batch.
        return self.forward_backward_update(x_batch, y_batch, batch_idx=i)

    def fit(self, *args, train_dataloaders, **kwargs) -> None:
        """
            Training starts

            Arguments
           ----------
           args:tuple
           (BASEKGE,)
           kwargs:Tuple
               empty dictionary
           Returns
           -------
           batch loss (float)
       """
        assert len(args) == 1
        model, = args
        self.model = model
        self.model.to(self.device)
        self.train_dataloaders = train_dataloaders
        self.loss_function = model.loss_function
        self.optimizer = self.model.configure_optimizers()
        self.training_step = self.model.training_step
        # (1) Start running callbacks
        self.on_fit_start(self, self.model)

        print(f'NumOfDataPoints:{len(self.train_dataloaders.dataset)} '
              f'| NumOfEpochs:{self.attributes.max_epochs} '
              f'| LearningRate:{self.model.learning_rate} '
              f'| BatchSize:{self.train_dataloaders.batch_size} '
              f'| EpochBatchsize:{len(train_dataloaders)}')

        for epoch in (tqdm_bar := tqdm(range(self.attributes.max_epochs))):
            self.model._current_epoch = epoch
            self.on_train_epoch_start(self, self.model)
            epoch_loss = 0
            i = 0
            construct_mini_batch_time = None
            batch: list
            for i, batch in enumerate(self.train_dataloaders):
                # (1) Extract Input and Outputs and set them on the dice
                x_batch, y_batch = self.extract_input_outputs_set_device(batch)
                start_time = time.time()
                if construct_mini_batch_time:
                    construct_mini_batch_time = start_time - construct_mini_batch_time
                # (2) Forward-Backward-Update.
                batch_loss = self._run_batch(i, x_batch, y_batch)
                epoch_loss += batch_loss
                tqdm_bar.set_description_str(f"Epoch:{epoch + 1}")
                if i>0:
                    tqdm_bar.set_postfix_str(f"loss_step={batch_loss:.5f}, loss_epoch={epoch_loss/i:.5f}")
                else:
                    tqdm_bar.set_postfix_str(f"loss_step={batch_loss:.5f}, loss_epoch={batch_loss:.5f}")
            avg_epoch_loss = epoch_loss / len(self.train_dataloaders)
            """
            # Autobatch Finder: Double the current batch size if memory allows and repeat this process at mast 5 times.
            if self.attributes.auto_batch_finder and psutil.virtual_memory().percent < 30.0 and counter < 5:
                self.train_dataloaders = DataLoader(dataset=self.train_dataloaders.dataset,
                                                    batch_size=self.train_dataloaders.batch_size
                                                               + self.train_dataloaders.batch_size,
                                                    shuffle=True, collate_fn=self.train_dataloaders.dataset.collate_fn,
                                                    num_workers=self.train_dataloaders.num_workers,
                                                    persistent_workers=False)
                print(
                    f'NumOfDataPoints:{len(self.train_dataloaders.dataset)} '
                    f'| NumOfEpochs:{self.attributes.max_epochs} '
                    f'| LearningRate:{self.model.learning_rate} '
                    f'| BatchSize:{self.train_dataloaders.batch_size} '
                    f'| EpochBatchsize:{len(train_dataloaders)}')
                counter += 1
            """
            self.model.loss_history.append(avg_epoch_loss)
            self.on_train_epoch_end(self, self.model)
        self.on_fit_end(self, self.model)

    def forward_backward_update(self, x_batch: torch.Tensor, y_batch: torch.Tensor, batch_idx: int = -1) -> torch.Tensor:
        """
            Compute forward, loss, backward, and parameter update

            Arguments
           ----------
           x_batch:(torch.Tensor) mini-batch inputs
           y_batch:(torch.Tensor) mini-batch outputs
           batch_idx:(int) index of the current mini-batch within the epoch

           Returns
           -------
           batch loss (float)
       """
        batch_loss = self.training_step(batch=(x_batch, y_batch))
        batch_loss.backward()
        self._log_gradients(batch_loss=batch_loss, batch_idx=batch_idx)
        self.optimizer.step()
        return batch_loss.item()

    def _log_gradients(self, batch_loss: torch.Tensor, batch_idx: int) -> None:
        """
        Print gradient diagnostics after backward() but before optimizer.step().

        - Every batch: total L2 grad norm across all parameters, plus loss value
          and counts of NaN / Inf in the gradient.
        - Batch 0 of each epoch: per-parameter breakdown (norm, min, max, mean,
          fraction of exactly-zero entries, NaN/Inf counts). This catches losses
          whose gradient vanishes everywhere (saturating clamps), explodes
          (NaN/Inf -> weights become NaN -> MRR ~ 0), or is identically zero for
          some submodule (e.g. relation embeddings never updated).
        """
        epoch = getattr(self.model, "_current_epoch", -1)
        loss_name = type(getattr(self.model, "loss", None)).__name__

        total_sq = 0.0
        total_nan = 0
        total_inf = 0
        total_params_with_grad = 0
        per_param_rows = []

        for name, param in self.model.named_parameters():
            if param.grad is None:
                if batch_idx == 0:
                    per_param_rows.append(f"  {name}: grad=None (no gradient flow)")
                continue
            g = param.grad.detach()
            # Use float32 reductions for numerical stability under AMP/bf16.
            g_f = g.float()
            nan_count = int(torch.isnan(g_f).sum().item())
            inf_count = int(torch.isinf(g_f).sum().item())
            # Replace NaN/Inf with 0 just for the norm so one bad param doesn't poison the total.
            g_finite = torch.nan_to_num(g_f, nan=0.0, posinf=0.0, neginf=0.0)
            sq = float((g_finite ** 2).sum().item())
            total_sq += sq
            total_nan += nan_count
            total_inf += inf_count
            total_params_with_grad += 1

            if batch_idx == 0:
                norm = sq ** 0.5
                gmin = float(g_finite.min().item())
                gmax = float(g_finite.max().item())
                gmean = float(g_finite.mean().item())
                zero_frac = float((g_f == 0).float().mean().item())
                per_param_rows.append(
                    f"  {name}: norm={norm:.3e} min={gmin:+.3e} max={gmax:+.3e} "
                    f"mean={gmean:+.3e} zero_frac={zero_frac:.3f} nan={nan_count} inf={inf_count}"
                )

        total_norm = total_sq ** 0.5
        loss_val = float(batch_loss.detach().item())

        if batch_idx == 0:
            print(f"\n[grad] epoch={epoch} batch=0 loss_fn={loss_name} loss={loss_val:.6f} "
                  f"total_norm={total_norm:.3e} nan={total_nan} inf={total_inf} "
                  f"params_with_grad={total_params_with_grad}")
            for row in per_param_rows:
                print(row)
        else:
            print(f"[grad] epoch={epoch} batch={batch_idx} loss={loss_val:.6f} "
                  f"total_norm={total_norm:.3e} nan={total_nan} inf={total_inf}")

    def extract_input_outputs_set_device(self, batch: list) -> Tuple:
        """
            Construct inputs and outputs from a batch of inputs with outputs From a batch of inputs and put

            Arguments
           ----------
           batch: (list) mini-batch inputs on CPU

           Returns
           -------
           (tuple) mini-batch on select device
       """
        if len(batch) == 2:
            x_batch, y_batch = batch

            if isinstance(x_batch, tuple):
                # Triple and Byte
                return x_batch, y_batch
            else:
                # (1) NegSample: x is a triple, y is a float
                x_batch, y_batch = batch
                return x_batch.to(self.device), y_batch.to(self.device)
        elif len(batch) == 3:
            x_batch, y_idx_batch, y_batch, = batch
            x_batch, y_idx_batch, y_batch = x_batch.to(self.device), y_idx_batch.to(self.device), y_batch.to(
                self.device)
            return (x_batch, y_idx_batch), y_batch
        else:
            print(len(batch))
            print("Unexpected batch shape..")
            raise RuntimeError
