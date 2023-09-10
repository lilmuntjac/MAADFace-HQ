from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

@dataclass
class Visualizer:
    pred_type: str
    fairness_matrix: str
    out_dir: str
    attr_list: list[str] | None = None
    marker: str = '.'
    markersize: int = 1
    annotation_gap: int = 5
    
    def resolve_fairness(self, stats):
        def get_binary_stats_per_epoch(stats_per_epoch):
            # the stats should have shape [A, 8]
            group_1_tp, group_1_fp, group_1_fn, group_1_tn = [stats_per_epoch[:,i] for i in range(0, 4)]
            group_2_tp, group_2_fp, group_2_fn, group_2_tn = [stats_per_epoch[:,i] for i in range(4, 8)]
            # Accuracy
            total_accuracy = (group_1_tp+group_1_tn+group_2_tp+group_2_tn)/(group_1_tp+group_1_fp+group_1_fn+group_1_tn+group_2_tp+group_2_fp+group_2_fn+group_2_tn)
            group_1_accuracy = (group_1_tp+group_1_tn)/(group_1_tp+group_1_fp+group_1_fn+group_1_tn)
            group_2_accuracy = (group_2_tp+group_2_tn)/(group_2_tp+group_2_fp+group_2_fn+group_2_tn)
            # Fairness
            group_1_tpr = group_1_tp/(group_1_tp+group_1_fn)
            group_2_tpr = group_2_tp/(group_2_tp+group_2_fn)
            group_1_tnr = group_1_tn/(group_1_fp+group_1_tn)
            group_2_tnr = group_2_tn/(group_2_fp+group_2_tn)
            equality_of_opportunity = abs(group_1_tpr-group_2_tpr) # (0, 1)
            equality_of_opportunity[np.isnan(equality_of_opportunity)] = 0
            equalized_odds = abs(group_1_tpr-group_2_tpr) + abs(group_1_tnr-group_2_tnr) # (0, 2)
            equalized_odds[np.isnan(equalized_odds)] = 0
            return {'total_accuracy': np.expand_dims(total_accuracy, axis=0),
                    'group_1_accuracy': np.expand_dims(group_1_accuracy, axis=0),
                    'group_2_accuracy': np.expand_dims(group_2_accuracy, axis=0),
                    'equality_of_opportunity': np.expand_dims(equality_of_opportunity, axis=0),
                    'equalized_odds': np.expand_dims(equalized_odds, axis=0)}
        def get_categorical_stats_per_epoch(stats_per_epoch):
            # the stats should have shape [A, 4]
            group_1_correct, group_1_wrong, group_2_correct, group_2_wrong = [stats_per_epoch[:,i] for i in range(0, 4)]
            group_1_acc = group_1_correct/(group_1_correct+group_1_wrong)
            group_2_acc = group_2_correct/(group_2_correct+group_2_wrong)
            total_acc = (group_1_correct+group_2_correct)/(group_1_correct+group_1_wrong+group_2_correct+group_2_wrong)
            accuracy_difference = abs(group_1_acc-group_2_acc)
            return {"group_1_accuracy": np.expand_dims(group_1_acc, axis=0), "group_2_accuracy": np.expand_dims(group_2_acc, axis=0), 
                    "total_accuracy": np.expand_dims(total_acc, axis=0), "accuracy_difference": np.expand_dims(accuracy_difference, axis=0)}
        match self.pred_type:
            case "binary":
                # the stats should have shape [N, A, 8]
                stats_dict = {'total_accuracy': np.array([]), 'group_1_accuracy': np.array([]), 'group_2_accuracy': np.array([]),
                              'equality_of_opportunity': np.array([]), 'equalized_odds': np.array([])}
                for stats_per_epoch in stats:
                    d = get_binary_stats_per_epoch(stats_per_epoch)
                    for key, values in stats_dict.items():
                        stats_dict.update({key: np.concatenate((values, d[key]), axis=0) if values.size > 0 else d[key]})
            case "categorical":
                # the stats should have shape [N, A, 4]
                stats_dict = {'total_accuracy': np.array([]), 'group_1_accuracy': np.array([]), 'group_2_accuracy': np.array([]), 'accuracy_difference': np.array([])}
                for stats_per_epoch in stats:
                    d = get_categorical_stats_per_epoch(stats_per_epoch)
                    for key, values in stats_dict.items():
                        stats_dict.update({key: np.concatenate((values, d[key]), axis=0) if values.size > 0 else d[key]})
            case _:
                assert False, "unknown prediction types"
        return stats_dict
    
    def draw_status_per_attribute(self, val_stats, train_stats=np.array([]), length=None,):
        def to_dict_key(str):
            match str:
                case "equality of opportunity":
                    assert self.pred_type == "binary", f'prediction type and fairness matrix mismatch'
                    return "equality_of_opportunity"
                case "equalized odds":
                    assert self.pred_type == "binary", f'prediction type and fairness matrix mismatch'
                    return "equalized_odds"
                case "accuracy difference":
                    assert self.pred_type == "categorical", f'prediction type and fairness matrix mismatch'
                    return "accuracy_difference"
                case _:
                    assert False, f'unrecognized fairness criteria'
        # one image per attribute
        for attr_idx, attr_name in enumerate(self.attr_list):
            fig_root = Path(self.out_dir)
            fig_root.mkdir(parents=True, exist_ok=True)
            fig_path = fig_root / (attr_name + '.png')
            val_stats_dict = self.resolve_fairness(val_stats)
            if train_stats.size > 0:
                train_stats_dict = self.resolve_fairness(train_stats)
            # draw the triple chart
            fig, axs  = plt.subplots(1,3, figsize=(24,8), layout='constrained')
            epoch_axis = np.linspace(0, length-1, length) if length else np.linspace(0, val_stats.shape[0]-1, val_stats.shape[0])
            # left chart - accuracy  / epoch
            val_tacc, = axs[0].plot(epoch_axis, val_stats_dict['total_accuracy'][:epoch_axis.shape[0],attr_idx], 
                                    marker=self.marker, markersize=self.markersize)
            val_g1acc, = axs[0].plot(epoch_axis, val_stats_dict['group_1_accuracy'][:epoch_axis.shape[0],attr_idx], 
                                    marker=self.marker, markersize=self.markersize)
            val_g2acc, = axs[0].plot(epoch_axis, val_stats_dict['group_2_accuracy'][:epoch_axis.shape[0],attr_idx], 
                                    marker=self.marker, markersize=self.markersize)
            for pt in range(1, epoch_axis.shape[0], self.annotation_gap):
                axs[0].annotate(f"{pt} \n{val_stats_dict['total_accuracy'][pt, attr_idx]:.2%}",
                                (epoch_axis[pt], val_stats_dict['total_accuracy'][pt, attr_idx]))
            axs[0].set_xlabel('Epochs')
            axs[0].set_ylabel('Accuracy')
            axs[0].set_box_aspect(1)
            axs[0].set_ylim([0.0, 1.0])
            if train_stats.size > 0:
                train_tacc, = axs[0].plot(epoch_axis, train_stats_dict['total_accuracy'][:epoch_axis.shape[0],attr_idx], 
                                         marker=self.marker, markersize=self.markersize)
                axs[0].legend((train_tacc, val_tacc, val_g1acc, val_g2acc), 
                              ('Training Acc.', 'Validation Acc.', 'Validation Group 1 Acc.', 'Validation Group 2 Acc.'), loc='lower right')
            else:
                axs[0].legend((val_tacc, val_g1acc, val_g2acc), 
                              ('Validation Acc.', 'Validation Group 1 Acc.', 'Validation Group 2 Acc.'), loc='lower right')
            # middle chart - fairness / epoch
            val_fairness, = axs[1].plot(epoch_axis, val_stats_dict[to_dict_key(self.fairness_matrix)][:epoch_axis.shape[0],attr_idx], 
                                                marker=self.marker, markersize=self.markersize)
            axs[1].set_xlabel('Epochs')
            axs[1].set_ylabel('Fairness, (lower the better)')
            axs[1].set_box_aspect(1)
            axs[1].set_ylim([0.0, 1.0])
            if train_stats.size > 0:
                train_fairness, = axs[1].plot(epoch_axis, train_stats_dict[to_dict_key(self.fairness_matrix)][:epoch_axis.shape[0],attr_idx], 
                                                    marker=self.marker, markersize=self.markersize)
                axs[1].legend((train_fairness, val_fairness), 
                              (f'Training {self.fairness_matrix}', f'Validation {self.fairness_matrix}',), loc='upper right')
            else:
                axs[1].legend((val_fairness,), (f'Validation {self.fairness_matrix}',), loc='upper right')
            # right chart - accuracy / fairness
            right, = axs[2].plot(list(map(lambda x: 1.0-x, val_stats_dict[to_dict_key(self.fairness_matrix)][:epoch_axis.shape[0],attr_idx])), 
                               val_stats_dict['total_accuracy'][:epoch_axis.shape[0],attr_idx], marker=self.marker, markersize=self.markersize)
            accuracy_change = val_stats_dict['total_accuracy'][:,attr_idx]-val_stats_dict['total_accuracy'][0,attr_idx]
            score_list = (accuracy_change+(1.0-val_stats_dict[to_dict_key(self.fairness_matrix)][:,attr_idx])).tolist()
            best_epoch = score_list.index(max(score_list))
            for pt in range(1, epoch_axis.shape[0], self.annotation_gap):
                axs[2].annotate(f"{pt} \n{1.0-val_stats_dict[to_dict_key(self.fairness_matrix)][pt,attr_idx]:.2%}",
                                (1.0-val_stats_dict[to_dict_key(self.fairness_matrix)][pt, attr_idx], 
                                 val_stats_dict['total_accuracy'][pt, attr_idx]))
            axs[2].set_title(f'Best epoch: {best_epoch}   Score: {score_list[best_epoch]:.4f} ')
            axs[2].set_xlabel(f'{self.fairness_matrix}')
            axs[2].set_ylabel('Total Accuracy')
            axs[2].set_box_aspect(1)
            axs[2].set_xlim([0.0, 1.0])
            axs[2].set_ylim([0.0, 1.0])
            fig.savefig(fig_path,)
            plt.close(fig)
            # print the best epoch performance
            print(f'==== {attr_name} ====')
            if train_stats.size > 0:
                print(f'Training:')
                print(f'    Group 1 Acc.: {train_stats_dict["group_1_accuracy"][best_epoch,attr_idx]:.4f}')
                print(f'    Group 2 Acc.: {train_stats_dict["group_2_accuracy"][best_epoch,attr_idx]:.4f}')
                print(f'    Total   Acc.: {train_stats_dict["total_accuracy"][best_epoch,attr_idx]:.4f}')
                if self.pred_type == "binary":
                    print(f'        Equality of opportunity: {train_stats_dict["equality_of_opportunity"][best_epoch,attr_idx]:.4f}')
                    print(f'        Equalized odds: {train_stats_dict["equalized_odds"][best_epoch,attr_idx]:.4f}')
                elif self.pred_type == "categorical":
                    print(f'        Accuracy difference: {train_stats_dict["accuracy_difference"][best_epoch,attr_idx]:.4f}')
            print(f'Validation:')
            print(f'    Group 1 Acc.: {val_stats_dict["group_1_accuracy"][best_epoch,attr_idx]:.4f}')
            print(f'    Group 2 Acc.: {val_stats_dict["group_2_accuracy"][best_epoch,attr_idx]:.4f}')
            print(f'    Total   Acc.: {val_stats_dict["total_accuracy"][best_epoch,attr_idx]:.4f}')
            if self.pred_type == "binary":
                print(f'        Equality of opportunity: {val_stats_dict["equality_of_opportunity"][best_epoch,attr_idx]:.4f}')
                print(f'        Equalized odds: {val_stats_dict["equalized_odds"][best_epoch,attr_idx]:.4f}')
            elif self.pred_type == "categorical":
                print(f'        Accuracy difference: {val_stats_dict["accuracy_difference"][best_epoch,attr_idx]:.4f}')
            print(f'')
            
def main(args):

    time_start = time.perf_counter()
    if args.train_stats:
        train_stats_path = Path(args.train_stats)
        train_stats = np.load(train_stats_path)
    else:
        train_stats=None

    val_stats_path = Path(args.val_stats)
    val_stats = np.load(val_stats_path) # in shape (N, A, 8) for binary/ (N, A, 4) for categorical
    print(f"The stats contain data from {val_stats.shape[0]} epochs, {val_stats.shape[1]} attributes.")
    assert len(args.attr_list) == val_stats.shape[1], "attributes list and stats not in the same shape"
    visualizer = Visualizer(pred_type=args.pred_type, fairness_matrix=args.fairness_matrix, out_dir=args.out_dir, attr_list=args.attr_list)
    visualizer.draw_status_per_attribute(val_stats, train_stats, )

    time_end = time.perf_counter()
    print(f'done in {(time_end-time_start)/60:.4f} mins')

    pass

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Show status")
    parser.add_argument("--val-stats", type=str, help="path to the validation stats, should be a .npy file")
    parser.add_argument("--train-stats", type=str, default=None, help="path to the training stats, should be a .npy file")
    parser.add_argument("-l", "--length", type=int, default=None, help="number of epochs shown in graph, start from 0")

    parser.add_argument("--pred-type", type=str, help="model prediction type, binary or categorical, saved in different format")
    parser.add_argument("--fairness-matrix", default="accuracy difference", help="how to measure fairness")
    parser.add_argument("--attr-list", type=str, nargs='+', help="attributes name predicted by model")


    parser.add_argument("-o", "--out-dir", type=str, help="output folder for all graph output")

    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)