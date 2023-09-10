from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# all the attribute from the dataset
attributes49 = ['Filename','Identity','Male','Young','Middle_Aged','Senior','Asian','White','Black',
                'Rosy_Cheeks','Shiny_Skin','Bald','Wavy_Hair','Receding_Hairline','Bangs','Sideburns','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair',
                'No_Beard','Mustache','5_o_Clock_Shadow','Goatee','Oval_Face','Square_Face','Round_Face','Double_Chin','High_Cheekbones','Chubby',
                'Obstructed_Forehead','Fully_Visible_Forehead','Brown_Eyes','Bags_Under_Eyes','Bushy_Eyebrows','Arched_Eyebrows',
                'Mouth_Closed','Smiling','Big_Lips','Big_Nose','Pointy_Nose','Heavy_Makeup',
                'Wearing_Hat','Wearing_Earrings','Wearing_Necktie','Wearing_Lipstick','No_Eyewear','Eyeglasses','Attractive']

cell_def = ['TP', 'FN', 'FP', 'TN']

def main(args):

    # helper function
    def resolve_binary_fairness(confusion_matrixs):
        # confusion matrix should be a numpy array in shape [8]
        # confusion matrix that belong to 2 groups
        group_1_tp, group_1_fp, group_1_fn, group_1_tn = confusion_matrixs[0:4]
        group_2_tp, group_2_fp, group_2_fn, group_2_tn = confusion_matrixs[4:8]
        #
        total_accuracy = (group_1_tp+group_1_tn+group_2_tp+group_2_tn)/sum(confusion_matrixs)
        group_1_accuracy = (group_1_tp+group_1_tn)/sum(confusion_matrixs[0:4])
        group_2_accuracy = (group_2_tp+group_2_tn)/sum(confusion_matrixs[4:8])
        # deal with the matrix that will cause NaN in the fairness criteria
        if group_1_tp+group_1_fn==0 or group_1_fp+group_1_tn==0 or \
           group_2_tp+group_2_fn==0 or group_2_fp+group_2_tn==0:
            equality_of_opportunity = -1
            equalized_odds = -1
        else:
            group_1_tpr = group_1_tp/(group_1_tp+group_1_fp)
            group_2_tpr = group_2_tp/(group_2_tp+group_2_fp)
            group_1_tnr = group_1_tn/(group_1_fn+group_1_tn)
            group_2_tnr = group_2_tn/(group_2_fn+group_2_tn)
            equality_of_opportunity = abs(group_1_tpr-group_2_tpr) # (0, 1)
            equalized_odds = abs(group_1_tpr-group_2_tpr) + abs(group_1_tnr-group_2_tnr) # (0, 2)
        return {'total_accuracy': total_accuracy,
                'group_1_accuracy': group_1_accuracy,
                'group_2_accuracy': group_2_accuracy,
                'equality_of_opportunity': equality_of_opportunity,
                'equalized_odds': equalized_odds}
    
    def draw_confusion_matrixs(attr_name, confusion_matrixs, root_folder='./'):
        # draw the confusion matrix for 2 groups and show their fairness status
        fig_root = Path(root_folder)
        fig_root.mkdir(parents=True, exist_ok=True)
        fig_path = fig_root / (attr_name + '.png')
        stats_dict = resolve_binary_fairness(confusion_matrixs)
        # draw the attributes status card
        fig, axs  = plt.subplots(1,2, figsize=(8,4), layout='constrained')
        fig.suptitle(attr_name+f'   Total accuracy: {stats_dict["total_accuracy"]:.2%} \
                     \nEquality of opportunity: {stats_dict["equality_of_opportunity"]:.2%}, Equalized odds: {stats_dict["equalized_odds"]:.2%}',
                     ha='center')
        
        left_fig = axs[0].imshow(confusion_matrixs[0:4].reshape(2,2)/sum(confusion_matrixs[0:4]), 
                                 cmap=plt.cm.cool, vmin=0, vmax=1)
        axs[0].set_title(f'Male accuracy: {stats_dict["group_1_accuracy"]:.2%}')
        axs[0].set_xticks([0, 1])
        axs[0].set_yticks([0, 1])
        axs[0].set_xlabel('Label')
        axs[0].set_ylabel('Prediction')
        axs[0].set_xticklabels(['Positive', 'Negative'])
        axs[0].set_yticklabels(['Positive', 'Negative'], rotation=90, va='center')
        
        right_fig=axs[1].imshow(confusion_matrixs[4:8].reshape(2,2)/sum(confusion_matrixs[4:8]), 
                                cmap=plt.cm.cool, vmin=0, vmax=1)
        axs[1].set_title(f'Female accuracy: {stats_dict["group_2_accuracy"]:.2%}')
        axs[1].set_xticks([0, 1])
        axs[1].set_yticks([0, 1])
        axs[1].set_xlabel('Label')
        axs[1].set_ylabel('Prediction')
        axs[1].set_xticklabels(['Positive', 'Negative'])
        axs[1].set_yticklabels(['Positive', 'Negative'], rotation=90, va='center')

        # value on the cell
        for i in range(4):
            axs[0].text(x=i%2, y=i//2, s=f'{cell_def[i]}\n{confusion_matrixs[i]/sum(confusion_matrixs[0:4]):.2%}', 
                        va='center', ha='center', size='xx-large')
            axs[1].text(x=i%2, y=i//2, s=f'{cell_def[i]}\n{confusion_matrixs[i+4]/sum(confusion_matrixs[4:8]):.2%}',  
                        va='center', ha='center', size='xx-large')
        
        # add a color bar
        fig.colorbar(left_fig, ax=axs[0],)
        fig.colorbar(right_fig, ax=axs[1],)

        # fig.tight_layout()
        fig.savefig(fig_path,)
        plt.close(fig)

    # load the stats
    stats_path = Path(args.stats)
    stats = np.load(stats_path)
    print(f"The stats contain data from {stats.shape[0]} epochs, {stats.shape[1]} attributes.")
    stats = stats[args.epoch,:,:]
    # get pair of confsion matirx
    assert len(args.attr_list) == stats.shape[0], "attributes list and stats not in the same shape"
    for attr in range(stats.shape[0]):
        draw_confusion_matrixs(args.attr_list[attr], stats[attr,:],  root_folder=args.out_dir)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Show stats per attributes")
    parser.add_argument("--stats", type=str, help="file path to the stats .npy file")
    parser.add_argument("-e", "--epoch", type=int, help="the epoch to get the confusion matrix on")
    parser.add_argument("-o", "--out-dir", type=str, help="output folder for attributes confusion matrix")

    parser.add_argument("--attr-list", type=str, nargs='+', help="attributes name predicted by model")

    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)