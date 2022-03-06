"""Extracts logits or probabilities. Currently not integrated into
main.py
"""

from typing import Dict
import csv
import torch
import pathlib
import dgl  # type: ignore
import collections

import dataset
import util.collect_dataset as col_dat
import utils

# python.exe .\main.py --mode train
# --saved_model_dir .\results\model_tree__classes_104__emb_dim_40__lstm_dim_200__fc_depth_3__label_smoothing_0_05__lr_0_002__prune_uniq_True
# --learning_rate 0.0001
# --batch_size 16
# --train_samples_epoch 52000
# --val_samples_epoch 10400
# --test_samples 52000
# --small_train_samples 100
# --epoch 17

# Should use the same args as in training for reproduction,
# but don't want to wait until it loads the whole data set.
ds_args = dataset.DataArgs(
    flatten=False,
    take_top10=False,
    training_weight = 8,
    validation_weight = 2,
    test_weight = 1,
    train_samples = 1,
    val_samples = 1,
    test_samples = 52000,
    small_train_samples = 1,
    batch_size=16,
    drop_large_threshold_tokens=400,
    max_per_class=None
)

_, _, _, dl_test = dataset.get_datasets(ds_args)

MODEL_PATH = pathlib.Path('results/'
                          'model_tree__classes_104__emb_dim_40'
                          '__lstm_dim_200__fc_depth_3'
                          '__label_smoothing_0_05__lr_0_0'
                          '__prune_uniq_True')
model, _ = utils.try_build_and_load_model(MODEL_PATH)
utils.load_model(model, MODEL_PATH, epoch=17)

output_file = csv.writer(
    open(str(MODEL_PATH / 'wrong_answer_logits.csv'), 'w')
)
output_file.writerow(
    ['true_class'] + [f'p_{i}' for i in range(104)])

wrong_classes : Dict[int, int] = collections.defaultdict(int) 

for batch in dl_test:
    with torch.no_grad():
        progs = dgl.batch([prog for prog, _ in batch]).to(model.device)
        targets = [cls for _, cls in batch]
        logits_t = model(progs)
        probabs = torch.nn.Softmax(dim=1)(
            torch.squeeze(logits_t)
        )
                
        predictions = torch.argmax(probabs, dim=1).cpu().numpy()
        probabs = list(probabs.cpu().numpy())
                
        for true_label, prediction, probab in zip(targets, predictions, probabs):
            if true_label != prediction:
                output_file.writerow([int(true_label)] + list(probab))
                wrong_classes[int(true_label)] += 1
                #print(true_label)
                #print(prediction)
                #print(probab)
                #sys.exit()

print(wrong_classes)
                      
        
        
        
