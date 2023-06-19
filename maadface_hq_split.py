import time
from pathlib import Path
import pandas as pd

def main(args):
    csv_all, csv_train, csv_test = Path(args.in_csv), Path(args.train_csv), Path(args.test_csv)
    print(f'Try split all the labels into train and test set by images available in the folders.')
    attribute_all = pd.read_csv(csv_all, sep=',', 
                                dtype={'Filename': 'str',
                                       'Identity': 'str', 'Male': int,
                                       'Young': int, 'Middle_Aged': int, 'Senior': int,
                                       'Asian': int, 'White': int, 'Black': int,
                                       'Rosy_Cheeks': int,
                                       'Shiny_Skin': int,
                                       'Bald': int, 'Wavy_Hair': int, 'Receding_Hairline': int, 'Bangs': int, 'Sideburns': int,
                                       'Black_Hair': int, 'Blond_Hair': int, 'Brown_Hair': int, 'Gray_Hair': int,
                                       'No_Beard': int, 'Mustache': int, '5_o_Clock_Shadow': int, 'Goatee': int,
                                       'Oval_Face': int, 'Square_Face': int, 'Round_Face': int,
                                       'Double_Chin': int,
                                       'High_Cheekbones': int,
                                       'Chubby': int,
                                       'Obstructed_Forehead': int, 'Fully_Visible_Forehead': int,
                                       'Brown_Eyes': int,
                                       'Bags_Under_Eyes': int,
                                       'Bushy_Eyebrows': int, 'Arched_Eyebrows': int,
                                       'Mouth_Closed': int,
                                       'Smiling': int,
                                       'Big_Lips': int,
                                       'Big_Nose': int,
                                       'Pointy_Nose': int,
                                       'Heavy_Makeup': int,
                                       'Wearing_Hat': int, 'Wearing_Earrings': int, 'Wearing_Necktie': int, 'Wearing_Lipstick': int,
                                       'No_Eyewear': int, 'Eyeglasses': int,
                                       'Attractive': int})
    # df = pd.DataFrame(columns=['Filename','Identity','Male','Young','Middle_Aged','Senior','Asian','White','Black',
    #                            'Rosy_Cheeks','Shiny_Skin','Bald','Wavy_Hair','Receding_Hairline','Bangs','Sideburns','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair',
    #                            'No_Beard','Mustache','5_o_Clock_Shadow','Goatee','Oval_Face','Square_Face','Round_Face','Double_Chin','High_Cheekbones','Chubby',
    #                            'Obstructed_Forehead','Fully_Visible_Forehead','Brown_Eyes','Bags_Under_Eyes','Bushy_Eyebrows','Arched_Eyebrows',
    #                            'Mouth_Closed','Smiling','Big_Lips','Big_Nose','Pointy_Nose','Heavy_Makeup',
    #                            'Wearing_Hat','Wearing_Earrings','Wearing_Necktie','Wearing_Lipstick','No_Eyewear','Eyeglasses','Attractive'])
    start_time = time.time()
    # train set
    train_list = list(Path(args.train_folder).glob('*/*.jpg'))
    train_filename = [p.parent.name+'/'+p.name for p in train_list]
    train_df = attribute_all.loc[attribute_all['Filename'].isin(train_filename)]
    train_df.to_csv(csv_train, index=False)
    train_time = time.time()-start_time
    print(f'train csv done in {train_time/60:.4f} mins with {len(train_df)} records.')

    start_time = time.time()
    # test set
    test_list = list(Path(args.test_folder).glob('*/*.jpg'))
    test_filename = [p.parent.name+'/'+p.name for p in test_list]
    test_df = attribute_all.loc[attribute_all['Filename'].isin(test_filename)]
    test_df.to_csv(csv_test, index=False)
    test_time = time.time()-start_time
    print(f'test csv done in {test_time/60:.4f} mins with {len(test_df)} records.')

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Split the MAADFace dataset by available image inside folders.")
    # 
    parser.add_argument("-i", "--in-csv", default='/tmp2/dataset/MAADFace_HQ/MAAD_Face.csv', type=str, help='original csv for MAADFace dataset')
    parser.add_argument("--train-csv", default='/tmp2/dataset/MAADFace_HQ/MAADFace_HQ_train.csv', type=str)
    parser.add_argument("--test-csv", default='/tmp2/dataset/MAADFace_HQ/MAADFace_HQ_test.csv', type=str)
    parser.add_argument("--train-folder", default='/tmp2/dataset/MAADFace_HQ/train', type=str)
    parser.add_argument("--test-folder", default="/tmp2/dataset/MAADFace_HQ/test", type=str)
    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)