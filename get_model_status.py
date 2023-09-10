from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    marker="."
    markersize=1

    # helper function
    def resolve_binary_fairness(confusion_matrixs):
        # confusion matrix should be a numpy array in shape [N, 8]
        total_accuracy_list, group_1_accuracy_list, group_2_accuracy_list = list(), list(), list()
        equality_of_opportunity_list, equalized_odds_list = list(), list()
        for epoch in range(confusion_matrixs.shape[0]):
            group_1_tp, group_1_fp, group_1_fn, group_1_tn = confusion_matrixs[epoch, 0:4]
            group_2_tp, group_2_fp, group_2_fn, group_2_tn = confusion_matrixs[epoch, 4:8]
            total_accuracy = (group_1_tp+group_1_tn+group_2_tp+group_2_tn)/sum(confusion_matrixs[epoch])
            group_1_accuracy = (group_1_tp+group_1_tn)/sum(confusion_matrixs[epoch, 0:4])
            group_2_accuracy = (group_2_tp+group_2_tn)/sum(confusion_matrixs[epoch, 4:8])
            # deal with the matrix that will cause NaN in the fairness criteria
            if group_1_tp+group_1_fn==0 or group_1_fp+group_1_tn==0 or \
               group_2_tp+group_2_fn==0 or group_2_fp+group_2_tn==0:
                equality_of_opportunity = -1
                equalized_odds = -1
            else:
                group_1_tpr = group_1_tp/(group_1_tp+group_1_fn)
                group_2_tpr = group_2_tp/(group_2_tp+group_2_fn)
                group_1_tnr = group_1_tn/(group_1_fp+group_1_tn)
                group_2_tnr = group_2_tn/(group_2_fp+group_2_tn)
                equality_of_opportunity = abs(group_1_tpr-group_2_tpr) # (0, 1)
                equalized_odds = abs(group_1_tpr-group_2_tpr) + abs(group_1_tnr-group_2_tnr) # (0, 2)
            total_accuracy_list.append(total_accuracy)
            group_1_accuracy_list.append(group_1_accuracy)
            group_2_accuracy_list.append(group_2_accuracy)
            equality_of_opportunity_list.append(equality_of_opportunity)
            equalized_odds_list.append(equalized_odds)
        # return the stats in dict format
        return {'total_accuracy': total_accuracy_list,
                'group_1_accuracy': group_1_accuracy_list,
                'group_2_accuracy': group_2_accuracy_list,
                'equality_of_opportunity': equality_of_opportunity_list,
                'equalized_odds': equalized_odds_list}
    
    def draw_binary_fairness_by_epochs(attr_name, confusion_matrixs, length=None, root_folder='./'):
        fig_root = Path(root_folder)
        fig_root.mkdir(parents=True, exist_ok=True)
        fig_path = fig_root / (attr_name + '.png')
        stats_dict = resolve_binary_fairness(confusion_matrixs) # list

        # draw the model status card
        fig, axs  = plt.subplots(1,2, figsize=(8,4), layout='constrained')
        fig.suptitle(attr_name)

        best_eqopp_epoch = stats_dict['equality_of_opportunity'].index(min(stats_dict['equality_of_opportunity']))
        left_fig = axs[0].plot(list(map(lambda x: 1.0-x, stats_dict['equality_of_opportunity'])), 
                               stats_dict['total_accuracy'], marker=marker, markersize=markersize)
        left_point = axs[0].scatter([1.0-stats_dict['equality_of_opportunity'][best_eqopp_epoch]],
                                    [stats_dict['total_accuracy'][best_eqopp_epoch]], )
        axs[0].set_title(f'Best epoch: {best_eqopp_epoch} - ')
        axs[0].set_xlabel('Equality of opportunity')
        axs[0].set_ylabel('Total Accuracy')
        axs[0].set_box_aspect(1)
        axs[0].set_xlim([0.0, 1.0])
        axs[0].set_ylim([0.0, 1.0])
        
        best_eqodd_epoch = stats_dict['equalized_odds'].index(min(stats_dict['equalized_odds']))
        right_fig= axs[1].plot(list(map(lambda x: 1.0-x, stats_dict['equalized_odds'])), 
                               stats_dict['total_accuracy'], marker=marker, markersize=markersize)
        right_point = axs[1].scatter([1.0-stats_dict['equalized_odds'][best_eqodd_epoch]],
                                     [stats_dict['total_accuracy'][best_eqodd_epoch]], )
        axs[1].set_title(f'Best epoch: {best_eqodd_epoch}')
        axs[1].set_xlabel('Equalized odds')
        axs[1].set_ylabel('Total Accuracy')
        axs[1].set_box_aspect(1)
        axs[1].set_xlim([0.5, 1.0])
        axs[1].set_ylim([0.0, 1.0])
        fig.savefig(fig_path,)
        plt.close(fig)

    def resolve_categorial_fairness(outcome_matrixs):
        # confusion matrix should be a numpy array in shape [N, 4]
        total_accuracy_list, group_1_accuracy_list, group_2_accuracy_list = list(), list(), list()
        accuracy_difference_list = list()
        for epoch in range(outcome_matrixs.shape[0]):
            group_1_correct, group_1_wrong, group_2_correct, group_2_wrong = [outcome_matrixs[epoch,i] for i in range(0, 4)]
            group_1_accuracy = group_1_correct/(group_1_correct+group_1_wrong)
            group_2_accuracy = group_2_correct/(group_2_correct+group_2_wrong)
            total_accuracy = (group_1_correct+group_2_correct)/(group_1_correct+group_1_wrong+group_2_correct+group_2_wrong)
            accuracy_difference = abs(group_1_accuracy-group_2_accuracy)
            total_accuracy_list.append(total_accuracy)
            group_1_accuracy_list.append(group_1_accuracy)
            group_2_accuracy_list.append(group_2_accuracy)
            accuracy_difference_list.append(accuracy_difference)
        # return the stats in dict format
        return {'total_accuracy': total_accuracy_list,
                'group_1_accuracy': group_1_accuracy_list,
                'group_2_accuracy': group_2_accuracy_list,
                'accuracy_difference': accuracy_difference_list}
    
    def draw_categorial_fairness_by_epochs(attr_name, outcome_matrixs, length=None, root_folder='./'):
        fig_root = Path(root_folder)
        fig_root.mkdir(parents=True, exist_ok=True)
        fig_path = fig_root / (attr_name + '.png')
        stats_dict = resolve_categorial_fairness(outcome_matrixs) # list

        # draw the model status card
        fig, axs  = plt.subplots(1,2, figsize=(8,4), layout='constrained')
        fig.suptitle(attr_name)

        best_accdiff_epoch = stats_dict['accuracy_difference'].index(min(stats_dict['accuracy_difference']))
        left_fig = axs[0].plot(list(map(lambda x: 1.0-x, stats_dict['accuracy_difference'])), 
                               stats_dict['total_accuracy'], marker=marker, markersize=markersize)
        left_point = axs[0].scatter([1.0-stats_dict['accuracy_difference'][best_accdiff_epoch]],
                                    [stats_dict['total_accuracy'][best_accdiff_epoch]], )
        axs[0].set_title(f'Best epoch: {best_accdiff_epoch}')
        axs[0].set_xlabel('Accuracy difference')
        axs[0].set_ylabel('Total Accuracy')
        axs[0].set_box_aspect(1)
        axs[0].set_xlim([0.5, 1.0])
        axs[0].set_ylim([0.5, 1.0])

        fig.savefig(fig_path,)
        plt.close(fig)

    stats_path = Path(args.stats)
    stats = np.load(stats_path) # in shape (N, A, 8) for binary/ (N, A, 4) for categorical
    print(f"The stats contain data from {stats.shape[0]} epochs, {stats.shape[1]} attributes.")
    assert len(args.attr_list) == stats.shape[1], "attributes list and stats not in the same shape"
    # make an image per attributes
    for attr_idx, attr in enumerate(args.attr_list):
        match args.pred_type:
            case "binary":
                draw_binary_fairness_by_epochs(attr, stats[:,attr_idx,:], root_folder=args.out_dir)
            case "categorical":
                draw_categorial_fairness_by_epochs(attr, stats[:,attr_idx,:], root_folder=args.out_dir)
            case _:
                assert False, "unknown model prediction type, must be binary or categorical"

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Show model tweaking status")
    parser.add_argument("--stats", type=str, help="file path to the stats .npy file")
    parser.add_argument("-o", "--out-dir", type=str, help="output folder for attributes confusion matrix")
    parser.add_argument("-l", "--length", type=int, default=None, help="number of epochs shown in graph")

    parser.add_argument("--attr-list", type=str, nargs='+', help="attributes name predicted by model")
    parser.add_argument("--pred-type", type=str, help="model prediction type, binary or categorical")

    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)