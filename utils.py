import json
import en_core_web_lg
from tqdm import tqdm

nlp = en_core_web_lg.load()

landmark_stop = []
with open("/home/joslin/cross-repr/landmark_stop.txt") as f1:
    for line in f1:
        landmark_stop.append(line.strip())


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

    
def construct_pairs(data, landmark_vocab=None):
    """ Construct landmarks and viewpoints pairs.
        :param filename: input data of json_file 

        :return: a dict with the constructed sentence and viewpoint pairs.
    """
    pair_list = []
    for each_d in tqdm(data):
        sent_list = each_d['new_instruction_list']
        chunk_view_list = each_d['chunk_view']
        viewpoint_id = each_d["path"]
        sent_view_pairs = [list(zip(*s_v)) for s_v in zip(*[sent_list, chunk_view_list])]  
        for sent_id, each_sent in enumerate(sent_view_pairs):         
            for sub_sent_id, each_sub_sent in enumerate(each_sent):
                pair_dict = {}
                viewpoint_list = viewpoint_id[each_sub_sent[1][0]:each_sub_sent[1][1]]
                pair_dict['pair_id'] = str(each_d['path_id'])+ "_" + str(sent_id) + "_" + str(sub_sent_id)
                pair_dict['scan'] = each_d['scan']
                pair_dict['sent'] = each_sub_sent[0]
                pair_dict['views'] = viewpoint_list
                tmp_sent = " ".join(each_sub_sent[0])
                pair_dict['landmarks'] = list(nlp(tmp_sent).noun_chunks)
                for each_land in pair_dict['landmarks']:
                    if each_land.text not in landmark_vocab and "right" not in each_land.text and "left" not in each_land.text and \
                       each_land.text not in landmark_stop :
                        landmark_vocab.append(each_land.text)
                pair_list.append(pair_dict)
    return pair_list, landmark_vocab



if __name__ == "__main__":
    path = "/home/joslin/cross-repr/Fine-Grained-R2R/new_data/new_FGR2R_"
    files = [path+ "train.json", path+"val_seen.json", path+"val_unseen.json"]
    landmark_vocab = []
    for each_file in files[:1]:
        with open(each_file) as f1:
            data = json.load(f1)
            # Construct the json files with paris.
            sent_view_pairs, landmark_vocab = construct_pairs(data, landmark_vocab=landmark_vocab)
             # Build landmark vocabs
    with open('/home/joslin/cross-repr/landmarks.txt','w') as f1:
        for land in landmark_vocab:
            f1.write(land+'\n')

    '''
    new_data = convert_instr_list(data, if_test=True)
    out_file = open("/home/joslin/cross-repr/Fine-Grained-R2R/new_data/new_FGR2R_test.json", "w")
    json.dump(new_data, out_file, indent = 6)
    ''' 
    
    

   
    



