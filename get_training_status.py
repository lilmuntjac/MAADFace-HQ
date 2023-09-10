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
    
    def draw_binary_model_status(attr_name, train_cm, val_cm, length=None, root_folder='./'):
        fig_root = Path(root_folder)
        fig_root.mkdir(parents=True, exist_ok=True)
        fig_path = fig_root / (attr_name + '.png')
        train_stats_dict = resolve_binary_fairness(train_cm) # list
        val_stats_dict = resolve_binary_fairness(val_cm) # list

        # draw the model status card
        fig, axs  = plt.subplots(1,2, figsize=(16,8), layout='constrained')
        fig.suptitle(attr_name)

        x_axis = np.linspace(0, length-1, length) if length else np.linspace(0, val_cm.shape[0]-1, val_cm.shape[0])
        # left fig.
        train_tacc, = axs[0].plot(x_axis, train_stats_dict['total_accuracy'][:x_axis.shape[0]], marker=marker, markersize=markersize)
        val_tacc, = axs[0].plot(x_axis, val_stats_dict['total_accuracy'][:x_axis.shape[0]], marker=marker, markersize=markersize)
        # annotation
        for pt in range(0, x_axis.shape[0], 5):
            axs[0].annotate(f"{pt} \n{val_stats_dict['total_accuracy'][pt]:.2%}", (x_axis[pt], val_stats_dict['total_accuracy'][pt]))
        
        
        axs[0].legend((train_tacc, val_tacc), ('Training Acc.', 'Validation Acc.',), loc='lower right')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_box_aspect(1)
        axs[0].set_ylim([0.5, 1.0])

        # right fig.
        train_eqopp, = axs[1].plot(x_axis, train_stats_dict['equality_of_opportunity'][:x_axis.shape[0]], marker=marker, markersize=markersize)
        train_eqodd, = axs[1].plot(x_axis, train_stats_dict['equalized_odds'][:x_axis.shape[0]], marker=marker, markersize=markersize)
        val_eqopp, = axs[1].plot(x_axis, val_stats_dict['equality_of_opportunity'][:x_axis.shape[0]], marker=marker, markersize=markersize)
        val_eqodd, = axs[1].plot(x_axis, val_stats_dict['equalized_odds'][:x_axis.shape[0]], marker=marker, markersize=markersize)
        axs[1].legend((train_eqopp, train_eqodd, val_eqopp, val_eqodd), ('Training equality of opportunity', 'Training equalized odds', 'Validation equality of opportunity', 'Validation equalized odds'), loc='upper right')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Fairness, (lower the better)')
        axs[1].set_box_aspect(1)
        axs[1].set_ylim([0.0, 1.0])

        fig.savefig(fig_path,)
        plt.close(fig)

    # print the epoch status on to the terminal
    def show_binary_model_status_by_epoch(attr_name, train_cm, val_cm, epoch):
        train_stats_dict = resolve_binary_fairness(train_cm) # list
        val_stats_dict = resolve_binary_fairness(val_cm) # list
        print(f'==== {attr_name} ====')
        print(f'Training:')
        print(f'    Group 1 Acc.: {train_stats_dict["group_1_accuracy"][epoch]:.4f}')
        print(f'    Group 2 Acc.: {train_stats_dict["group_2_accuracy"][epoch]:.4f}')
        print(f'    Total   Acc.: {train_stats_dict["total_accuracy"][epoch]:.4f}')
        print(f'        Equality of opportunity: {train_stats_dict["equality_of_opportunity"][epoch]:.4f}')
        print(f'        Equalized odds: {train_stats_dict["equalized_odds"][epoch]:.4f}')
        print(f'Validation:')
        print(f'    Group 1 Acc.: {val_stats_dict["group_1_accuracy"][epoch]:.4f}')
        print(f'    Group 2 Acc.: {val_stats_dict["group_2_accuracy"][epoch]:.4f}')
        print(f'    Total   Acc.: {val_stats_dict["total_accuracy"][epoch]:.4f}')
        print(f'        Equality of opportunity: {val_stats_dict["equality_of_opportunity"][epoch]:.4f}')
        print(f'        Equalized odds: {val_stats_dict["equalized_odds"][epoch]:.4f}')
        print(f'')

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

    def draw_categorial_model_status(attr_name, train_oc, val_oc, length=None, root_folder='./'):
        fig_root = Path(root_folder)
        fig_root.mkdir(parents=True, exist_ok=True)
        fig_path = fig_root / (attr_name + '.png')
        train_stats_dict = resolve_categorial_fairness(train_oc) # list
        val_stats_dict = resolve_categorial_fairness(val_oc) # list

        # draw the model status card
        fig, axs  = plt.subplots(1,2, figsize=(16,8), layout='constrained')
        fig.suptitle(attr_name)

        x_axis = np.linspace(0, length-1, length) if length else np.linspace(0, val_oc.shape[0]-1, val_oc.shape[0])
        # left fig.
        train_tacc, = axs[0].plot(x_axis, train_stats_dict['total_accuracy'][:x_axis.shape[0]], marker=marker, markersize=markersize)
        val_tacc, = axs[0].plot(x_axis, val_stats_dict['total_accuracy'][:x_axis.shape[0]], marker=marker, markersize=markersize)
        # annotation
        for pt in range(1, x_axis.shape[0], 2):
            axs[0].annotate(f"{pt} \n{val_stats_dict['total_accuracy'][pt]:.2%}", (x_axis[pt], val_stats_dict['total_accuracy'][pt]))
        
        
        axs[0].legend((train_tacc, val_tacc), ('Training Acc.', 'Validation Acc.',), loc='lower right')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_box_aspect(1)
        axs[0].set_ylim([0.5, 1.0])

        # right fig.
        train_accdiff, = axs[1].plot(x_axis, train_stats_dict['accuracy_difference'][:x_axis.shape[0]], marker=marker, markersize=markersize)
        val_accdiff, = axs[1].plot(x_axis, val_stats_dict['accuracy_difference'][:x_axis.shape[0]], marker=marker, markersize=markersize)
        axs[1].legend((train_accdiff, val_accdiff), ('Training accuracy difference', 'Validation accuracy difference'), loc='upper right')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Fairness, (lower the better)')
        axs[1].set_box_aspect(1)
        axs[1].set_ylim([0.0, 1.0])

        fig.savefig(fig_path,)
        plt.close(fig)
    
    # print the epoch status on to the terminal
    def show_categorial_model_status_by_epoch(attr_name, train_oc, val_oc, epoch):
        train_stats_dict = resolve_categorial_fairness(train_oc) # list
        val_stats_dict = resolve_categorial_fairness(val_oc) # list
        print(f'==== {attr_name} ====')
        print(f'Training:')
        print(f'    Group 1 Acc.: {train_stats_dict["group_1_accuracy"][epoch]:.4f}')
        print(f'    Group 2 Acc.: {train_stats_dict["group_2_accuracy"][epoch]:.4f}')
        print(f'    Total   Acc.: {train_stats_dict["total_accuracy"][epoch]:.4f}')
        print(f'        Accuracy difference: {train_stats_dict["accuracy_difference"][epoch]:.4f}')
        print(f'Validation:')
        print(f'    Group 1 Acc.: {val_stats_dict["group_1_accuracy"][epoch]:.4f}')
        print(f'    Group 2 Acc.: {val_stats_dict["group_2_accuracy"][epoch]:.4f}')
        print(f'    Total   Acc.: {val_stats_dict["total_accuracy"][epoch]:.4f}')
        print(f'        Accuracy difference: {val_stats_dict["accuracy_difference"][epoch]:.4f}')
        print(f'')

    train_stats_path, val_stats_path = Path(args.train_stats), Path(args.val_stats)
    train_stats = np.load(train_stats_path) # in shape (N, A, 8) for binary/ (N, A, 4) for categorical
    val_stats = np.load(val_stats_path)

    print(f"The stats contain data from {train_stats.shape[0]} epochs, {train_stats.shape[1]} attributes.")
    assert len(args.attr_list) == train_stats.shape[1], "attributes list and stats not in the same shape"
    assert train_stats.shape == val_stats.shape, "make sure the train stats match the val stats"
    # make an image per attributes
    for attr_idx, attr in enumerate(args.attr_list):
        match args.pred_type:
            case "binary":
                draw_binary_model_status(attr, train_stats[:,attr_idx,:], val_stats[:,attr_idx,:], length=args.length, root_folder=args.out_dir)
                show_binary_model_status_by_epoch(attr, train_stats[:,attr_idx,:], val_stats[:,attr_idx,:], args.epoch_shown)
            case "categorical":
                draw_categorial_model_status(attr, train_stats[:,attr_idx,:], val_stats[:,attr_idx,:], length=args.length, root_folder=args.out_dir)
                show_categorial_model_status_by_epoch(attr, train_stats[:,attr_idx,:], val_stats[:,attr_idx,:], args.epoch_shown)
            case _:
                assert False, "unknown model prediction type, must be binary or categorical"
    

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Show model tweaking status")
    parser.add_argument("--train-stats", type=str, help="file path to the train stats .npy file")
    parser.add_argument("--val-stats", type=str, help="file path to the validation stats .npy file")
    parser.add_argument("-o", "--out-dir", type=str, help="output folder for attributes confusion matrix")
    parser.add_argument("-l", "--length", type=int, default=None, help="number of epochs shown in graph")

    parser.add_argument("--attr-list", type=str, nargs='+', help="attributes name predicted by model")
    parser.add_argument("--pred-type", type=str, help="model prediction type, binary or categorical")

    parser.add_argument("-e", "--epoch-shown", type=int, help="specify the epoch to print to the terminal")

    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)