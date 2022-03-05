import torch
import pathlib
import ast
import pathlib

import dataset
import util.collect_dataset as col_dat
import utils
import linear_lstm_model as linear_model
import dgl_lstm_model as tree_model

def predict(model, program_dir):
    
    program_str = open(program_dir, 'r').read()
    
    ast_parsed = ast.parse(program_str)
    tree = col_dat.to_simple_tree(ast_parsed)
    dgl_tree = dataset.to_dgl_tree(str(tree))
    flat_prog = dataset.flatten_program(str(tree))

    take_top10 = model.num_classes == 10
    problems = sorted(col_dat.MOST_COMMON10
                      if take_top10
                      else col_dat.MOST_COMMON)

    is_linear = isinstance(model, linear_model.LinearLSTM)

    with torch.no_grad():
        if is_linear:
            batch_logits = model(torch.unsqueeze(flat_prog, 1),
                        torch.tensor([len(flat_prog)]))
        else:
            batch_logits = model(dgl_tree.to(model.device))
            
        batch_probs = list(torch.nn.Softmax(dim=0)(
            torch.squeeze(batch_logits)
        ).cpu().numpy())
        probs_labels = list(zip(batch_probs, problems))
        print()
        for probab, problem in reversed(
                sorted(probs_labels)[-3:]):
            print(f"P={probab:.3f}, problem={problem[0]}{problem[1]}")
            
        
if __name__ == "__main__":
    
    large_linear = (
        'results/' +
        'model_linear__classes_104__emb_dim_40__lstm_dim_200__fc_depth_3__label_smoothing_0_05__lr_0_0001')

    model, _ = utils.try_build_and_load_model(large_linear)

    for p in pathlib.Path('data/my/').glob('*.py'):
        print(f"\nPredicting {p}: ")
        predict(model, str(p))
