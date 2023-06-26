# Customize Train Template

Here, we present how to develop a new Train Template, and apply it into EduStudio. EduStudio provides the TrainTPL Protocol in ``EduStudio.edustudio.traintpl.basetraintpl.BaseTrainTPL`` (``BaseTrainTPL``).

## TrainTPL Protocol 

The TrainTPL Protocol is detailed in ``BaseTrainTPL``. The function to start the training process is ``BaseTrainTPL.start()``.

## TrainTPLs 

By inherenting the TrainTPL Protocol, EduStudio provides the class ``EduStudio.edustudio.traintpl.traintpl.gd_traintpl.GDTrainTPL``(``GDTrainTPL``) and ``EduStudio.edustudio.traintpl.edu_traintpl.EduTrainTPL``(``EduTrainTPL``), which are suitable for most gradient descent optimization-based models and most student evaluation models.  ``GDTrainTPL`` inherits ``BaseTrainTPL``  and rewrites ``start()``. The function to get optimizer according to the parameter ``default_cfg.optim`` is ``GDTrainTPL._get_optim()``. The function to obtain loaders of train, val, and test dataset is ``GDTrainTPL.build_loaders()``.  ``EduTrainTPL`` inherits ``GDTrainTPL`` and rewrites ``start()``. In the ``EduTrainTPL.start()``, the functions for each dataloader is ``EduTrainTPL.fit()`` .

## Develop a New TrainTPL in EduStudio

If the developed model needs more complex training method, then one can inherent ``BaseTrainTPL`` and revise the function ``start()``. One can also define the configuration of the new training template in the dictionary ``default_cfg``.  Similarly, one can inherent ``GDTrainTPL`` and ``EduTrainTPL`` and revise the ``start`` function and ``default_cfg`` dictionary.

Example
-------------------------
If you need to modify TrainTPl in the student assessment model so that only ``main_loss`` is used after a certain epoch, then you just need to inherit ``EduTrainTPL``, set the ``epoch_to_change`` parameter in ``default_cfg``.

```python
from .edu_traintpl import EduTrainTPL
class NewTrainTPL(EduTrainTPL):
    default_cfg = {
        'epoch_to_change': 10,
    }

def __init__(self, cfg: UnifyConfig):
    super().__init__(cfg)
```

Then, one can rewrite the ``fit`` function.


```python
def fit(self, train_loader, val_loader):
    ...
    for epoch in range(self.trainfmt_cfg['epoch_num']):
        ...
        for batch_id, batch_dict in enumerate(
                tqdm(train_loader, ncols=self.frame_cfg['TQDM_NCOLS'], desc="[EPOCH={:03d}]".format(epoch + 1))
        ):
            batch_dict = self.batch_dict2device(batch_dict)
            loss_dict = self.model.get_loss_dict(**batch_dict)

            ############ This part is revised ############

            if epoch < self.traintpl_cfg['epoch_to_change']:
                loss = torch.hstack([i for i in loss_dict.values() if i is not None]).sum()
            else:
                loss = loss_dict['main_loss']

            ############ This part is revised ############
            ...
        ...
    ...
```

The complete code of example is detailed as follows.

```python
from .edu_traintpl import EduTrainTPL
class NewTrainTPL(EduTrainTPL):
    default_cfg = {
        'epoch_to_change': 10,
    }

def __init__(self, cfg: UnifyConfig):
    super().__init__(cfg)
	
def fit(self, train_loader, val_loader):
    self.model.train()
    self.optimizer = self._get_optim()
    self.callback_list.on_train_begin()
    for epoch in range(self.trainfmt_cfg['epoch_num']):
        self.callback_list.on_epoch_begin(epoch + 1)
        logs = defaultdict(lambda: np.full((len(train_loader),), np.nan, dtype=np.float32))
        for batch_id, batch_dict in enumerate(
                tqdm(train_loader, ncols=self.frame_cfg['TQDM_NCOLS'], desc="[EPOCH={:03d}]".format(epoch + 1))
        ):
            batch_dict = self.batch_dict2device(batch_dict)
            loss_dict = self.model.get_loss_dict(**batch_dict)

            ############ This part is revised ############

            if epoch < self.traintpl_cfg['epoch_to_change']:
                loss = torch.hstack([i for i in loss_dict.values() if i is not None]).sum()
            else:
                loss = loss_dict['main_loss']

            ############ This part is revised ############

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            for k in loss_dict: logs[k][batch_id] = loss_dict[k].item() if loss_dict[k] is not None else np.nan

        for name in logs: logs[name] = float(np.nanmean(logs[name]))

        if val_loader is not None:
            val_metrics = self.evaluate(val_loader)
            logs.update({f"{metric}": val_metrics[metric] for metric in val_metrics})

        self.callback_list.on_epoch_end(epoch + 1, logs=logs)
        if self.model.share_callback_dict.get('stop_training', False):
            break

    self.callback_list.on_train_end()
```
