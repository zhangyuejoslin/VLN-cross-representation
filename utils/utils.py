import json
import en_core_web_lg
from tqdm import tqdm
import numpy as np
import random


nlp = en_core_web_lg.load()
candidate_path = '/VL/space/zhan1624/exploration/R2R-EnvDrop/candidate.npy'
candidate_dict = np.load(candidate_path, allow_pickle=True).item()
image_feat = np.load('/egr/research-hlr/joslin/Matterdata/v1/scans/img_features/mycsvfile1.npy', allow_pickle=True).item()

with open("/VL/space/zhan1624/VLN-cross-representation/utils/landmark_stop.txt") as f1:
    landmark_stop = [line.strip() for line in f1]


def convert_instr_list(data, test=False):
    """ The instructions in the original file are string of list. This function is to convert string to list.

        :param filename: input data of json_file 
        :param test: the format of test file is different, and it is without the key of chunk_view

        :return: the json data with the instruction list rather than string
    """
    for each_d in data:
        sent_str = each_d['new_instructions']
        if not test:
            chunk_view_list = each_d['chunk_view']
        sent_str = sent_str.lstrip("[[[").rstrip("]]]")
        sent_list = sent_str.split("]], [[")
        new_sent = []
        for sent_id, sent in enumerate(sent_list):
            tmp_sub_sent = []
            sent = sent.split("], [")
            for each_sub in sent:
                tokens = each_sub.split(',')
                tmp_sub_sent.append([each_token.strip().lstrip("\'").rstrip("\'") for each_token in tokens])
            new_sent.append(tmp_sub_sent)
            if not test:
                assert len(tmp_sub_sent) == len(chunk_view_list[sent_id])
        if not test:
            assert len(new_sent) == len(chunk_view_list)
        each_d['new_instruction_list'] = new_sent
    return data

def get_viewheading(scan_id, view_list, positive=False):
    """ Add heading information to viewpoint list.
        :param scan_id
        :param viewpoint_list: the ground-truth viewpoint list.

        :return: the viewpoint list with heading information.
    """
    def get_heading(heading):
        headingIndex = {0:'0.0', 1:'0.5235987755982988', 2:'1.0471975511965976', 3:'1.5707963267948966', 4:'2.0943951023931953', 5:'2.6179938779914944',
                         6:'3.141592653589793', 7:'3.6651914291880923', 8:'4.1887902047863905', 9:'4.71238898038469', 10:'5.235987755982989', 
                         11:'5.759586531581287'}
        if heading < 12:
            return headingIndex[heading*0] + "_" + "-0.5235987755982988"
        elif heading >= 12 and heading < 24:
            return headingIndex[heading - 12*1] + "_" + "0.0"
        else:
            return headingIndex[heading - 12*2] + "_" + "0.5235987755982988"

    new_view_list = []
    for view_id, view in enumerate(view_list):
        if view_id < len(view_list)-1:
            if positive:
                new_view = view_list[view_id+1]
                heading = candidate_dict[scan_id][view_list[view_id]][new_view]
                new_view_list.append((view, heading, new_view))
            else:
                candidates = candidate_dict[scan_id][view_list[view_id]]
                neg_candidates = list(set(candidates.keys()) - set([view_list[view_id+1]]))
                if not neg_candidates: # Cases that no candidates left except the ground-truth viewpoints
                    new_view_list.append((view, None, None))
                    continue
                new_view = neg_candidates[random.randint(0, len(neg_candidates))-1] # randint(a,b) -> a<=x<=b
                heading = candidate_dict[scan_id][view_list[view_id]][new_view] # Select a negative randomly
                new_view_list.append((view, heading, new_view))
    assert len(new_view_list) == len(view_list)-1
    return new_view_list
    
def construct_pairs(data, positive=False):
    """ Construct landmarks and viewpoints pairs.
        :param filename: input data of json_file 

        :return: a dict with the constructed sentence and viewpoint pairs.
    """
    pair_list = []
    for each_d in tqdm(data):
        sent_list = each_d['new_instruction_list']
        chunk_view_list = each_d['chunk_view']
        if positive:
            viewpoint_id = get_viewheading(each_d["scan"], each_d["path"], positive) # Get viewpoint list with heading information.
        else:
            viewpoint_id = get_viewheading(each_d["scan"], each_d["path"], positive)
        sent_view_pairs = [list(zip(*s_v)) for s_v in zip(*[sent_list, chunk_view_list])]  
        for sent_id, each_sent in enumerate(sent_view_pairs):         
            for sub_sent_id, each_sub_sent in enumerate(each_sent):
                pair_dict = {}
                landmarks = []
                viewpoint_list = viewpoint_id[each_sub_sent[1][0]-1: each_sub_sent[1][1]-1]
                if positive:
                    # format of "pair_id":[r2r_example_id, instr_id, sub_instr_id, target, view_id]
                    # target: {positive:0, negative:1}
                    pair_dict['pair_id'] = str(each_d['path_id'])+ "_" + str(sent_id) + "_" + str(sub_sent_id)+ "_" + "0"
                    pair_dict['target'] = "positive"
                else:
                    pair_dict['pair_id'] = str(each_d['path_id'])+ "_" + str(sent_id) + "_" + str(sub_sent_id)+ "_" + "1"
                    pair_dict['target'] = "negative"
                pair_dict['views'] = viewpoint_list
                pair_dict['scan'] = each_d['scan']
                tmp_sent = " ".join(each_sub_sent[0])
                for each_land in list(nlp(tmp_sent).noun_chunks):
                    if "right" not in each_land.text and "left" not in each_land.text and \
                       each_land.text not in landmark_stop :
                        landmarks.append(each_land.text)
                pair_dict['landmarks'] = landmarks
                if pair_dict['landmarks'] and pair_dict['views']:
                    pair_list.append(pair_dict)

    return pair_list


def get_img_feat(scanid, view):
    if view[1]:
        img_feat = image_feat[scanid][view[0]][view[1]]
        img_id = scanid + "_" + view[0] + "_" + str(view[1])
        return (img_id, img_feat['text'], img_feat['boxes'])
    else:
        return None


if __name__ == "__main__":

    # step1: convert sub-instruction from string to list.
    '''
    new_data = convert_instr_list(data, if_test=True)
    out_file = open("/home/joslin/cross-repr/Fine-Grained-R2R/new_data/new_FGR2R_test.json", "w")
    json.dump(new_data, out_file, indent = 6)
    ''' 

    
    # step2: Construct the positive and negative paris.     
    '''
    path = "/VL/space/zhan1624/VLN-cross-representation/new_data/new_FGR2R_"
    files = [path+ "train.json", path+"val_seen.json", path+"val_unseen.json"]
    each_file = files[0]
    
    with open(each_file) as f1:
        data = json.load(f1)
        sent_view_pairs_postive = construct_pairs(data, positive = True)
        sent_view_pairs_negative = construct_pairs(data, positive = False)
        np.save("/VL/space/zhan1624/VLN-cross-representation/pairs/positive.npy", sent_view_pairs_postive)
        np.save("/VL/space/zhan1624/VLN-cross-representation/pairs/negative.npy", sent_view_pairs_negative)
    '''

    # step3: Construct the detailed example
    positive_examples = np.load("/VL/space/zhan1624/VLN-cross-representation/pairs/positive.npy", allow_pickle=True).tolist()
    negative_examples = np.load("/VL/space/zhan1624/VLN-cross-representation/pairs/negative.npy", allow_pickle=True).tolist()
    all_examples = positive_examples + negative_examples 
    new_examples = []
    for each_pos in tqdm(all_examples):
        for view_id, each_view in enumerate(each_pos['views']):
            tmp_dict = {}
            img_feat= get_img_feat(each_pos["scan"], each_view)
            if not img_feat: 
                continue
            img_id, img_labels, img_boxes = img_feat
            tmp_dict['pair_id'] = each_pos['pair_id'] + "_" + str(view_id)
            tmp_dict['labels'] = img_labels
            tmp_dict['boxes'] = img_boxes
            tmp_dict['image_id'] = img_id
            tmp_dict["landmark"] = each_pos["landmarks"]
            tmp_dict["target"] = each_pos["target"]
            new_examples.append(tmp_dict)
    np.save("/VL/space/zhan1624/VLN-cross-representation/pairs/examples.npy", new_examples, allow_pickle=True)