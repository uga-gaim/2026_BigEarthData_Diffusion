import os
import pandas as pd

def opinionated_checkpoint_selection(ckpt_dir, margin=2.):
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    
    ckpt_epoch = [int(f.split('=')[1].split('-')[0]) for f in ckpt_files]
    ckpt_loss = [float(f.split('=')[-1].rstrip('.ckpt')) for f in ckpt_files]
    ckpt_type = [f.split('_')[0].lstrip('Monitor').lower() for f in ckpt_files]
    df = pd.DataFrame({'file': ckpt_files, 'epoch': ckpt_epoch, 'loss': ckpt_loss, 'type': ckpt_type})
    best_of_val = df[df.type == 'valloss'].sort_values(by='loss').iloc[:1]
    df = pd.concat([df[df.type=='epoch'], best_of_val], axis=0, ignore_index=True)
    df = df[df.loss <= best_of_val.loss.item() * (1 + margin)]
    selected_ckpt = df.sort_values(by='epoch', ascending=False).iloc[0].file
    return selected_ckpt