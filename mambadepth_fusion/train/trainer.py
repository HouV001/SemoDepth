import csv
import os
import random
import time
import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from mambadepth_fusion.data import ZJUDataset
from mambadepth_fusion.model import MambaFusionDepth
from mambadepth_fusion.utils import eval as eval_utils


def train(train_image_paths, train_radar_paths, train_gt_paths, train_sparse_gt_paths,
          train_mono_paths=None, val_image_paths=None, val_radar_paths=None, val_sparse_gt_paths=None,
          val_mono_paths=None, eval_image_paths=None, eval_radar_paths=None, eval_sparse_gt_paths=None,
          eval_mono_paths=None, learning_rate=0.0001, num_epochs=60, batch_size=8, n_step_per_summary=50,
          n_step_per_checkpoint=2000, min_depth=0.5, max_depth=80.0, lambda_grad=0.5, lambda_linear=1.0,
          lambda_log=1.0, lambda_sparse=1.0, augment=True, pretrained_mae=None, mono_dropout_prob=0.5,
          resume=None, checkpoint_dirpath='./checkpoints/mambadepth', restore_path='', n_threads=16,
          dataset_class=None, seed=None):
    use_mono_depth = train_mono_paths is not None
    os.makedirs(checkpoint_dirpath, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    log_path = os.path.join(checkpoint_dirpath, 'results.txt')
    train_csv_path = os.path.join(checkpoint_dirpath, 'training_log.csv')
    eval_csv_path = os.path.join(checkpoint_dirpath, 'test_results.csv')
    val_csv_path = os.path.join(checkpoint_dirpath, 'val_results.csv')
    config_path = os.path.join(checkpoint_dirpath, 'config.txt')
    config = {'learning_rate': learning_rate, 'num_epochs': num_epochs, 'batch_size': batch_size,
              'min_depth': min_depth, 'max_depth': max_depth, 'lambda_grad': lambda_grad,
              'lambda_linear': lambda_linear, 'lambda_log': lambda_log, 'lambda_sparse': lambda_sparse,
              'augment': augment, 'n_train': len(train_gt_paths), 'pretrained_mae': pretrained_mae,
              'use_mono_depth': use_mono_depth,
              'mono_dropout_prob': mono_dropout_prob if use_mono_depth else 0.0, 'seed': seed}
    with open(config_path, 'w') as f:
        for k, v in config.items():
            f.write(f'{k}: {v}\n')
    train_csv_fields = ['step', 'epoch', 'loss', 'loss_log', 'loss_linear', 'loss_grad', 'loss_sparse', 'mae_mm', 'lr']
    eval_csv_fields = ['epoch', 'step']
    for max_d in [50, 70, 80]:
        for m in ['mae', 'rmse', 'imae', 'irmse', 'delta1']:
            eval_csv_fields.append(f'{m}_{max_d}')
    if not resume:
        with open(train_csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=train_csv_fields).writeheader()
        with open(eval_csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=eval_csv_fields).writeheader()
        with open(val_csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=eval_csv_fields).writeheader()
    DatasetCls = dataset_class or ZJUDataset
    if dataset_class is not None:
        ds_kwargs = dict(augment=augment, max_depth=max_depth)
        if use_mono_depth:
            ds_kwargs['mono_paths'] = train_mono_paths
            ds_kwargs['mono_dropout_prob'] = mono_dropout_prob
        train_dataset = DatasetCls(train_image_paths, train_radar_paths, train_gt_paths, train_sparse_gt_paths, **ds_kwargs)
    else:
        train_dataset = ZJUDataset(train_image_paths, train_radar_paths, train_gt_paths, train_sparse_gt_paths, augment=augment, max_depth=max_depth)

    def _worker_init_fn(worker_id):
        if seed is not None:
            s = seed + worker_id
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
    loader_generator = None
    if seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_threads, drop_last=True, persistent_workers=True, prefetch_factor=4, pin_memory=True, worker_init_fn=_worker_init_fn, generator=loader_generator)
    model = MambaFusionDepth(min_depth=min_depth, max_depth=max_depth, lambda_grad=lambda_grad,
                             lambda_linear=lambda_linear, lambda_log=lambda_log, lambda_sparse=lambda_sparse,
                             use_mono_depth=use_mono_depth).to(device)
    if restore_path and os.path.exists(restore_path):
        model.load(restore_path)
        print(f'Loaded weights from {restore_path}')
    if pretrained_mae and os.path.exists(pretrained_mae) and hasattr(model, 'load_pretrained_mae'):
        model.load_pretrained_mae(pretrained_mae)
    n_total = sum((p.numel() for p in model.parameters()))
    n_enc = sum((p.numel() for p in model.get_encoder_params()))
    n_dec = sum((p.numel() for p in model.get_decoder_params()))
    print(f'MambaFusionDepth: {n_total / 1000000.0:.1f}M total ({n_enc / 1000000.0:.1f}M encoder + {n_dec / 1000000.0:.1f}M decoder)')
    print(f'LR: {learning_rate:.0e} | augment={augment}')
    if use_mono_depth:
        if mono_dropout_prob > 0.0:
            mode_desc = f'TacoDepth-style (p={mono_dropout_prob:.2f}, single checkpoint supports both plug-in and independent eval)'
        else:
            mode_desc = 'RadarCam-Depth-style (plug-in only, D* always required)'
        print(f'plug-in: ON | mono_dropout={mono_dropout_prob:.2f} → {mode_desc}')
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"Mixed precision: {('bf16' if use_amp else 'disabled')}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: max(1.0 - epoch // 10 * 0.1, 0.5))
    writer = SummaryWriter(os.path.join(checkpoint_dirpath, 'events'))
    step = 0
    start_epoch = 1
    t0 = time.time()
    best_mae_80 = float('inf')
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        step = ckpt['step']
        best_mae_80 = ckpt.get('best_mae_80', float('inf'))
        print(f'Resumed from {resume}, epoch {start_epoch}, step {step}, best_mae={best_mae_80:.0f}')
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_metrics = []
        for batch_data in train_loader:
            batch_data = [x.to(device, non_blocking=True) for x in batch_data]
            mono = None
            if use_mono_depth and len(batch_data) == 5:
                image, radar, gt, sparse_gt, mono = batch_data
            else:
                image, radar, gt, sparse_gt = batch_data[:4]
            step += 1
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                loss, ld = model.forward_train(image, radar, gt, sparse_gt=sparse_gt, mono_depth=mono)
            if loss.item() == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_metrics.append(ld)
            if step % n_step_per_summary == 0:
                lr_now = optimizer.param_groups[0]['lr']
                for k, v in ld.items():
                    writer.add_scalar(f'train/{k}', v, step)
                elapsed = (time.time() - t0) / 3600
                sparse_str = f" spar={ld['loss_sparse']:.4f}" if ld.get('loss_sparse', 0) > 0 else ''
                print(f"[Ep{epoch} Step{step}] loss={ld['loss']:.4f} log={ld['loss_log']:.4f} lin={ld['loss_linear']:.4f}{sparse_str} mae={ld['mae_mm']:.0f}mm [{elapsed:.1f}h]")
                row = {'step': step, 'epoch': epoch, 'lr': f'{lr_now:.6f}', **{k: f'{v:.6f}' for k, v in ld.items() if k in train_csv_fields}}
                with open(train_csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=train_csv_fields).writerow(row)
            if step % n_step_per_checkpoint == 0:
                torch.save({'epoch': epoch, 'step': step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_mae_80': best_mae_80}, os.path.join(checkpoint_dirpath, f'checkpoint-{step}.pth'))
                if val_sparse_gt_paths is not None:
                    val_results = evaluate(val_image_paths, val_radar_paths, val_sparse_gt_paths, model=model, device=device, dataset_class=dataset_class, max_depth=max_depth, mono_paths=val_mono_paths)
                    _log_eval_results(val_results, epoch, step, log_path, val_csv_path, eval_csv_fields, writer, prefix='VAL', tag='val')
                if eval_sparse_gt_paths is not None:
                    results = evaluate(eval_image_paths, eval_radar_paths, eval_sparse_gt_paths, model=model, device=device, dataset_class=dataset_class, max_depth=max_depth, mono_paths=eval_mono_paths)
                    _log_eval_results(results, epoch, step, log_path, eval_csv_path, eval_csv_fields, writer, prefix='TEST', tag='test')
                    mae_80 = results.get(80, {}).get('mae', float('inf'))
                    if mae_80 < best_mae_80:
                        best_mae_80 = mae_80
                        model.save(os.path.join(checkpoint_dirpath, 'model-best.pth'))
                        with open(log_path, 'a') as f:
                            f.write(f'  *** New best MAE_80={mae_80:.0f} ***\n')
                        print(f'  *** New best MAE_80={mae_80:.0f} ***')
        scheduler.step()
        if epoch_metrics:
            avg = {k: np.mean([m[k] for m in epoch_metrics]) for k in epoch_metrics[0]}
            print(f"--- Epoch {epoch}/{num_epochs} avg_loss={avg['loss']:.4f} avg_mae={avg['mae_mm']:.0f}mm")
    model.save(os.path.join(checkpoint_dirpath, f'model-{step}.pth'))
    model.save(os.path.join(checkpoint_dirpath, 'model-final.pth'))
    writer.close()
    with open(log_path, 'a') as f:
        f.write(f'\n=== Training complete ===\n')
        f.write(f'Best MAE_80: {best_mae_80:.0f}mm\n')
        f.write(f'Total time: {(time.time() - t0) / 3600:.1f}h\n')
    print(f'\nTraining complete. Best MAE_80={best_mae_80:.0f}mm')


def _log_eval_results(results, epoch, step, log_path, csv_path, csv_fields, writer, prefix, tag):
    row = {'epoch': epoch, 'step': step}
    with open(log_path, 'a') as f:
        f.write(f'\n=== Step {step}, Epoch {epoch} [{prefix}] ===\n')
    for max_d in [50, 70, 80]:
        r = results.get(max_d)
        if not r:
            continue
        print(f"  {prefix} {max_d}m: MAE={r['mae']:.0f} RMSE={r['rmse']:.0f} iMAE={r['imae']:.2f} iRMSE={r['irmse']:.2f} d1={r['delta1']:.4f}")
        with open(log_path, 'a') as f:
            f.write(f"  {max_d}m: MAE={r['mae']:.0f} RMSE={r['rmse']:.0f} iMAE={r['imae']:.2f} iRMSE={r['irmse']:.2f} d1={r['delta1']:.4f}\n")
        for m_name in ['mae', 'rmse', 'imae', 'irmse', 'delta1']:
            row[f'{m_name}_{max_d}'] = f'{r[m_name]:.4f}'
            writer.add_scalar(f'{tag}/{m_name}_{max_d}', r[m_name], step)
    with open(csv_path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=csv_fields).writerow(row)


def evaluate(image_paths, radar_paths, sparse_gt_paths, model, device=None, dataset_class=None,
             max_depth=None, mono_paths=None, default_to_max=True):
    if device is None:
        device = next(model.parameters()).device
    if max_depth is None and hasattr(model, 'max_d'):
        max_depth = float(model.max_d)
    min_eval = float(getattr(model, 'min_d', 0.5))
    use_mono = mono_paths is not None
    if dataset_class is not None:
        ds_kwargs = dict(augment=False, max_depth=max_depth)
        if use_mono:
            ds_kwargs['mono_paths'] = mono_paths
        dataset = dataset_class(image_paths, radar_paths, sparse_gt_paths, sparse_gt_paths, **ds_kwargs)
    else:
        dataset = ZJUDataset(image_paths, radar_paths, sparse_gt_paths, sparse_gt_paths, augment=False, max_depth=max_depth)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8, pin_memory=True)
    model.eval()
    range_metrics = {max_d: {'mae': [], 'rmse': [], 'imae': [], 'irmse': [], 'delta1': []} for max_d in [50, 70, 80]}
    for batch_data in loader:
        batch_data = [x.to(device, non_blocking=True) for x in batch_data]
        mono = None
        if use_mono and len(batch_data) == 5:
            image, radar, _, sparse_gt, mono = batch_data
        else:
            image, radar, _, sparse_gt = batch_data[:4]
        depth, _ = model.forward_test(image, radar, mono_depth=mono)
        B = depth.shape[0]
        for b in range(B):
            pred_np = depth[b].squeeze().cpu().numpy()
            gt_np = sparse_gt[b].squeeze().cpu().numpy()
            for max_d in [50, 70, 80]:
                if default_to_max:
                    p = np.where(np.isnan(pred_np), min_eval, pred_np)
                    p = np.where(np.isinf(p), max_d, p)
                    p = np.clip(p, min_eval, max_d)
                else:
                    p = pred_np
                valid = np.where((gt_np > 0) & (gt_np <= max_d))
                o, g = (p[valid], gt_np[valid])
                if len(o) == 0:
                    continue
                m = range_metrics[max_d]
                m['mae'].append(eval_utils.mean_abs_err(1000 * o, 1000 * g))
                m['rmse'].append(eval_utils.root_mean_sq_err(1000 * o, 1000 * g))
                m['imae'].append(eval_utils.inv_mean_abs_err(0.001 * o, 0.001 * g))
                m['irmse'].append(eval_utils.inv_root_mean_sq_err(0.001 * o, 0.001 * g))
                m['delta1'].append(eval_utils.thr_acc(o, g))
    results = {}
    for max_d in [50, 70, 80]:
        m = range_metrics[max_d]
        results[max_d] = {k: float(np.mean(v)) if len(v) > 0 else float('nan') for k, v in m.items()}
        results[max_d]['n_frames'] = len(m['mae'])
    model.train()
    return results
