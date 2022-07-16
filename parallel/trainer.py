import torch.nn
import os
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import optim
from utils.misc import *
import wandb


class ParallelTrainer(object):
    def __init__(self, args, config, model, criterion, optimizer = None, scheduler = None):
        self.args = args
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay) if optimizer is None else optimizer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=args.decay_factor, patience=args.decay_patience) if scheduler is None else scheduler
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.model = self.accelerator.prepare_model(self.model)
        self.optimizer = self.accelerator.prepare_optimizer(self.optimizer)
        if self.accelerator.is_local_main_process:
            if not os.path.exists(self.args.save_ckpt_dir):
                os.makedirs(self.args.save_ckpt_dir)

    def train(self, train_loader, eval_loader = None, test_loader = None):
        train_loader = self.accelerator.prepare_data_loader(train_loader)
        if self.args.enable_wandb and self.accelerator.is_local_main_process:
            wandb.init(
                name="skempi-gvp-mlp",
                group="geogvp",
                project='gvp',
                config=self.args,
                dir=os.path.join(self.args.wandb_log_dir),
                # mode="offline"
            )
        for epoch in range(self.args.epochs):
            self.model.train()
            train_loss = 0.0
            for step, batch in enumerate(train_loader):
                batch = recursive_to(batch, device=self.args.device)
                pred = self.model(batch)
                #print(pred, batch['ddG'])
                if self.args.mode == 'cla':
                    loss = self.criterion(pred, batch['ddG'])
                    #print(loss)
                elif self.args.mode == 'reg':
                    loss = self.criterion(pred, batch['ddG'].float())
                else:
                    mean, var = pred[:, 0], pred[:, 1]
                    var = torch.nn.functional.softplus(var) + 1e-6
                    loss = self.criterion(mean, batch['ddG'], var)
                loss = loss / self.args.gradient_accumulation
                #print("pred, target, loss", pred, batch['ddG'], loss)
                train_loss += loss

                self.accelerator.backward(loss)
                if step % self.args.gradient_accumulation == 0 or step == len(train_loader) - 1:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler is not None and self.args.lr_scheduler_type == 'Noam':
                        self.scheduler.step()
            # evaluation and lr decay
            if eval_loader is not None:
                val_loss = self.evaluate(eval_loader)
                if test_loader is not None:
                    test_loss, acc, spearman_coef = self.test(test_loader)
                    self.accelerator.print(
                        "{}-th epoch train loss={:.4f}, lr={}, val loss={:.4f}, test loss= {:.4f}, acc={:.4f}, spearman coef={:.4f}".
                            format(epoch, train_loss.item(), self.optimizer.param_groups[0]['lr'], val_loss.item(), test_loss.item(), acc.item(), spearman_coef))
                    if self.args.enable_wandb and self.accelerator.is_local_main_process:
                        metrics = {'train_loss': train_loss.item(), 'lr': self.optimizer.param_groups[0]['lr'],  'val_loss': val_loss.item(),
                                   'test_loss': test_loss.item(), 'test_acc': acc.item(), 'spearman':spearman_coef}
                        wandb.log(metrics)
                else: # test_loader is none
                    self.accelerator.print(
                        "{}-th epoch train loss={:.4f}, lr={}, val loss={:.4f}".
                            format(epoch, train_loss.item(), self.optimizer.param_groups[0]['lr'], val_loss.item()))
                    if self.args.enable_wandb and self.accelerator.is_local_main_process:
                        metrics = {'train_loss': train_loss.item(), 'lr': self.optimizer.param_groups[0]['lr'],  'val_loss': val_loss.item()}
                        wandb.log(metrics)

                if self.scheduler is not None and self.args.lr_scheduler_type == 'Plateau':
                    self.scheduler.step(val_loss)
            else: # eval_loader is None
                if test_loader is not None:
                    test_loss, acc, spearman_coef = self.test(test_loader)
                    self.accelerator.print(
                        "{}-th epoch train loss={:.4f}, lr={}, test loss= {:.4f}, acc={:.4f}, spearman coef={:.4f}".
                            format(epoch, train_loss.item(), self.optimizer.param_groups[0]['lr'], test_loss.item(), acc.item(), spearman_coef))
                    if self.args.enable_wandb and self.accelerator.is_local_main_process:
                        metrics = {'train_loss': train_loss.item(), 'lr': self.optimizer.param_groups[0]['lr'],
                                   'test_loss': test_loss.item(), 'test_acc': acc.item(), 'spearman':spearman_coef}
                        wandb.log(metrics)
                else: # test_loader is none
                    self.accelerator.print("{}-th epoch train loss={}, lr={}".format(epoch, train_loss.item(), self.optimizer.param_groups[0]['lr']))
                    if self.args.enable_wandb and self.accelerator.is_local_main_process:
                        metrics = {'train_loss': train_loss.item(), 'lr': self.optimizer.param_groups[0]['lr']}
                        wandb.log(metrics)
                    if self.scheduler is not None and self.args.lr_scheduler_type == 'Plateau':
                        self.scheduler.step(train_loss)

            torch.cuda.empty_cache()
            # model checkpoint
            if epoch % self.args.ckpt_freq == 0:
                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                model_name = "model_{}.pt".format(epoch)
                if self.accelerator.is_local_main_process:
                    save_checkpoint(self.config, unwrapped_model, os.path.join(self.args.save_ckpt_dir, model_name))
        if self.args.enable_wandb and self.accelerator.is_local_main_process:
            wandb.finish()

    def evaluate(self, eval_loader):
        self.model.eval()
        eval_loader = self.accelerator.prepare_data_loader(eval_loader)
        eval_loss = 0.0
        for batch in eval_loader:
            with torch.no_grad():
                batch = recursive_to(batch, device=self.args.device)
                pred = self.model(batch)
                if self.args.mode == 'cla':
                    loss = self.criterion(pred, batch['ddG'])
                elif self.args.mode == 'reg':
                    loss = self.criterion(pred, batch['ddG'].float())
                else:
                    mean, var = pred[:, 0], pred[:, 1]
                    var = torch.nn.functional.softplus(var) + 1e-6
                    loss = self.criterion(mean, batch['ddG'], var)
                eval_loss += loss

        return eval_loss

    def test(self, test_loader):
        self.model.eval()
        num_instances = len(test_loader)
        test_loader = self.accelerator.prepare_data_loader(test_loader)
        all_outputs = list()
        all_targets = list()
        all_signs = list()
        test_loss = 0.0
        for batch in test_loader:
            with torch.no_grad():
                batch = recursive_to(batch, device=self.args.device)
                pred = self.model(batch)
                if self.args.mode == 'cla':
                    loss = self.criterion(pred, batch['ddG'])
                elif self.args.mode == 'reg':
                    loss = self.criterion(pred, batch['ddG'].float())
                else:
                    mean, var = pred[:, 0], pred[:, 1]
                    var = torch.nn.functional.softplus(var) + 1e-6
                    loss = self.criterion(mean, batch['ddG'], var)
                test_loss += loss

            if self.args.mode == 'cla':
                pred = torch.nn.Softmax(dim=1)(pred)
                pred = torch.argmax(pred, dim=1)
                sign = (pred - 1.5) * (batch['ddG'] - 1.5)
                outputs = self.accelerator.gather(pred.contiguous())
                targets = self.accelerator.gather(batch['ddG'].contiguous())
            elif self.args.mode == 'reg':
                outputs = self.accelerator.gather(pred.contiguous())
                targets = self.accelerator.gather(batch['ddG'].float().contiguous())
                sign = pred * batch['ddG'].float()
            else:
                outputs = self.accelerator.gather(mean.contiguous())
                targets = self.accelerator.gather(batch['ddG'].float().contiguous())
                sign = mean * batch['ddG'].float()
            sign = self.accelerator.gather(sign.contiguous())
            all_outputs.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_signs.append(sign.detach().cpu())
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_signs = torch.cat(all_signs, dim=0)
        correct = torch.where(all_signs > 0, torch.ones_like(all_signs), torch.zeros_like(all_signs)).sum(-1)
        acc = correct / float(num_instances)
        all_outputs, all_targets = all_outputs.numpy().ravel(), all_targets.numpy().ravel()
        spearman_coef = spearman(y_pred=all_outputs, y_true=all_targets)
        return test_loss, acc, spearman_coef
